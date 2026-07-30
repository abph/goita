const VOICE_RECONNECT_DELAY_MS = 2000;
const SPEAKING_START_FRAMES = 2;
const SPEAKING_STOP_FRAMES = 8;
const SPEAKING_RMS_THRESHOLD = 0.045;

function validSeat(seat) {
  return ["A", "B", "C", "D"].includes(seat);
}

function websocketUrl(roomId, seat, clientId) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams({ seat, client_id: clientId });
  return `${protocol}//${window.location.host}/voice/${encodeURIComponent(roomId)}?${params}`;
}

class VoiceChatController {
  constructor(options = {}) {
    this.getContext = options.getContext || (() => ({}));
    this.onChange = options.onChange || (() => {});
    this.onHint = options.onHint || (() => {});
    this.onDuckingChange = options.onDuckingChange || (() => {});

    this.socket = null;
    this.localStream = null;
    this.joinContext = null;
    this.iceServers = [];
    this.peers = new Map();
    this.participants = new Map();
    this.speakingSeats = new Set();
    this.shouldReconnect = false;
    this.reconnectTimer = null;
    this.joining = false;
    this.joined = false;
    this.muted = true;
    this.localSpeaking = false;
    this.audioContext = null;
    this.analyser = null;
    this.levelFrame = 0;
    this.speakingFrames = 0;
    this.silentFrames = 0;
    this.ducking = false;
  }

  snapshot() {
    return {
      joining: this.joining,
      joined: this.joined,
      active: !!this.localStream,
      muted: this.muted,
      participants: Array.from(this.participants.values()),
      speakingSeats: Array.from(this.speakingSeats),
    };
  }

  emit() {
    const nextDucking = this.speakingSeats.size > 0;
    if (nextDucking !== this.ducking) {
      this.ducking = nextDucking;
      this.onDuckingChange(nextDucking);
    }
    this.onChange(this.snapshot());
  }

  contextIsEligible(context = this.getContext()) {
    return (
      context?.roomId === "debug"
      && context?.eligible === true
      && validSeat(context?.seat)
      && typeof context?.clientId === "string"
      && context.clientId.length > 0
    );
  }

  async join() {
    if (this.localStream || this.joining) return;
    const context = this.getContext();
    if (!this.contextIsEligible(context)) {
      this.onHint("ボイスチャットは、デバッグルームで着席している人だけ利用できます。");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) {
      this.onHint("このブラウザーではボイスチャットを利用できません。");
      return;
    }

    this.joining = true;
    this.muted = true;
    this.emit();
    try {
      const configParams = new URLSearchParams({
        seat: context.seat,
        client_id: context.clientId,
      });
      const configResponse = await fetch(
        `/games/${encodeURIComponent(context.roomId)}/voice/config?${configParams}`
      );
      if (!configResponse.ok) {
        throw new Error("voice config denied");
      }
      const config = await configResponse.json();
      this.iceServers = Array.isArray(config?.iceServers) ? config.iceServers : [];
      this.localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      this.localStream.getAudioTracks().forEach((track) => {
        track.enabled = false;
      });
      const latestContext = this.getContext();
      if (
        !this.contextIsEligible(latestContext)
        || latestContext.roomId !== context.roomId
        || latestContext.seat !== context.seat
        || latestContext.clientId !== context.clientId
      ) {
        throw new Error("voice context changed");
      }
      this.joinContext = {
        roomId: context.roomId,
        seat: context.seat,
        clientId: context.clientId,
      };
      this.shouldReconnect = true;
      this.startLevelDetection();
      this.connectSignalSocket();
    } catch (error) {
      console.warn("Voice chat join failed:", error);
      this.joining = false;
      this.shouldReconnect = false;
      this.stopLocalStream();
      this.onHint("マイクを利用できませんでした。ブラウザーのマイク許可を確認してください。");
      this.emit();
    }
  }

  async leave() {
    this.shouldReconnect = false;
    this.joining = false;
    this.joined = false;
    this.muted = true;
    this.setLocalSpeaking(false, false);
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      socket.onclose = null;
      try {
        socket.close(1000, "voice chat left");
      } catch (_error) {}
    }
    this.clearPeers();
    this.stopLevelDetection();
    this.stopLocalStream();
    this.joinContext = null;
    this.participants.clear();
    this.speakingSeats.clear();
    this.emit();
  }

  syncContext() {
    if (!this.localStream && !this.joining) {
      this.emit();
      return;
    }
    const context = this.getContext();
    const sameContext = (
      this.contextIsEligible(context)
      && context.roomId === this.joinContext?.roomId
      && context.seat === this.joinContext?.seat
      && context.clientId === this.joinContext?.clientId
    );
    if (!sameContext) this.leave();
  }

  toggleMute() {
    if (!this.localStream) return;
    this.muted = !this.muted;
    this.localStream.getAudioTracks().forEach((track) => {
      track.enabled = !this.muted;
    });
    if (this.muted) this.setLocalSpeaking(false, false);
    this.sendVoiceState();
    this.emit();
  }

  connectSignalSocket() {
    if (!this.shouldReconnect || !this.joinContext) return;
    if (
      this.socket
      && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const context = this.joinContext;
    const socket = new WebSocket(
      websocketUrl(context.roomId, context.seat, context.clientId)
    );
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket !== socket) return;
      this.joined = true;
      this.joining = false;
      this.sendVoiceState();
      this.emit();
    };
    socket.onmessage = (event) => {
      if (this.socket !== socket) return;
      try {
        Promise.resolve(this.handleMessage(JSON.parse(event.data))).catch((error) => {
          console.warn("Voice message failed:", error);
        });
      } catch (error) {
        console.warn("Voice message failed:", error);
      }
    };
    socket.onclose = (event) => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.joined = false;
      this.joining = false;
      this.clearPeers();
      this.participants.clear();
      this.speakingSeats.clear();
      if ([4001, 4002, 4403].includes(event.code)) {
        this.shouldReconnect = false;
        this.stopLevelDetection();
        this.stopLocalStream();
        this.joinContext = null;
        this.muted = true;
        this.onHint("ボイスチャットから退出しました。");
      }
      this.emit();
      if (this.shouldReconnect && this.contextStillMatches()) {
        this.reconnectTimer = window.setTimeout(() => {
          this.reconnectTimer = null;
          this.connectSignalSocket();
        }, VOICE_RECONNECT_DELAY_MS);
      }
    };
    socket.onerror = () => {
      if (this.socket === socket) this.onHint("ボイスチャットを再接続しています。");
    };
  }

  contextStillMatches() {
    const context = this.getContext();
    return (
      this.contextIsEligible(context)
      && context.roomId === this.joinContext?.roomId
      && context.seat === this.joinContext?.seat
      && context.clientId === this.joinContext?.clientId
    );
  }

  send(payload) {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(payload));
    return true;
  }

  sendSignal(type, target, data) {
    if (!validSeat(target)) return;
    this.send({ type, target, data });
  }

  sendVoiceState() {
    this.send({
      type: "voice_state",
      muted: this.muted,
      speaking: this.localSpeaking && !this.muted,
    });
  }

  async handleMessage(message) {
    if (!message || typeof message !== "object") return;
    if (message.type === "voice_roster") {
      await this.handleRoster(message.participants);
      return;
    }
    const sourceSeat = String(message.from || "").toUpperCase();
    if (!validSeat(sourceSeat) || sourceSeat === this.joinContext?.seat) return;
    if (message.type === "offer") {
      await this.acceptOffer(sourceSeat, message.data);
    } else if (message.type === "answer") {
      await this.acceptAnswer(sourceSeat, message.data);
    } else if (message.type === "ice") {
      await this.acceptIceCandidate(sourceSeat, message.data);
    }
  }

  async handleRoster(rawParticipants) {
    const nextParticipants = new Map();
    for (const raw of Array.isArray(rawParticipants) ? rawParticipants : []) {
      const seat = String(raw?.seat || "").toUpperCase();
      if (!validSeat(seat)) continue;
      nextParticipants.set(seat, {
        seat,
        muted: raw?.muted !== false,
        speaking: raw?.speaking === true,
      });
    }
    this.participants = nextParticipants;

    const ownSeat = this.joinContext?.seat;
    this.speakingSeats.clear();
    for (const participant of nextParticipants.values()) {
      const speaking = participant.seat === ownSeat
        ? this.localSpeaking && !this.muted
        : participant.speaking && !participant.muted;
      if (speaking) this.speakingSeats.add(participant.seat);
    }

    for (const peerSeat of Array.from(this.peers.keys())) {
      if (!nextParticipants.has(peerSeat)) this.removePeer(peerSeat);
    }
    for (const peerSeat of nextParticipants.keys()) {
      if (peerSeat === ownSeat) continue;
      const peer = this.ensurePeer(peerSeat);
      if (ownSeat < peerSeat && !peer.offerStarted) {
        peer.offerStarted = true;
        try {
          const offer = await peer.pc.createOffer();
          await peer.pc.setLocalDescription(offer);
          this.sendSignal("offer", peerSeat, peer.pc.localDescription);
        } catch (error) {
          console.warn("Voice offer failed:", error);
          peer.offerStarted = false;
        }
      }
    }
    this.emit();
  }

  ensurePeer(peerSeat) {
    const existing = this.peers.get(peerSeat);
    if (existing) return existing;

    const pc = new RTCPeerConnection({ iceServers: this.iceServers });
    const audio = document.createElement("audio");
    audio.autoplay = true;
    audio.playsInline = true;
    audio.hidden = true;
    audio.dataset.voiceSeat = peerSeat;
    document.getElementById("voiceAudioOutputs")?.appendChild(audio);

    this.localStream?.getTracks().forEach((track) => {
      pc.addTrack(track, this.localStream);
    });
    const peer = {
      pc,
      audio,
      offerStarted: false,
      pendingCandidates: [],
    };
    pc.onicecandidate = (event) => {
      if (event.candidate) this.sendSignal("ice", peerSeat, event.candidate);
    };
    pc.ontrack = (event) => {
      const [stream] = event.streams;
      audio.srcObject = stream || new MediaStream([event.track]);
      const playPromise = audio.play();
      if (playPromise?.catch) playPromise.catch(() => {});
    };
    pc.onconnectionstatechange = () => {
      if (["failed", "closed"].includes(pc.connectionState)) {
        this.removePeer(peerSeat);
      }
    };
    this.peers.set(peerSeat, peer);
    return peer;
  }

  async acceptOffer(peerSeat, description) {
    const peer = this.ensurePeer(peerSeat);
    await peer.pc.setRemoteDescription(description);
    await this.flushIceCandidates(peer);
    const answer = await peer.pc.createAnswer();
    await peer.pc.setLocalDescription(answer);
    this.sendSignal("answer", peerSeat, peer.pc.localDescription);
  }

  async acceptAnswer(peerSeat, description) {
    const peer = this.ensurePeer(peerSeat);
    await peer.pc.setRemoteDescription(description);
    await this.flushIceCandidates(peer);
  }

  async acceptIceCandidate(peerSeat, candidate) {
    if (!candidate) return;
    const peer = this.ensurePeer(peerSeat);
    if (!peer.pc.remoteDescription) {
      peer.pendingCandidates.push(candidate);
      return;
    }
    await peer.pc.addIceCandidate(candidate);
  }

  async flushIceCandidates(peer) {
    while (peer.pendingCandidates.length) {
      await peer.pc.addIceCandidate(peer.pendingCandidates.shift());
    }
  }

  removePeer(peerSeat) {
    const peer = this.peers.get(peerSeat);
    if (!peer) return;
    this.peers.delete(peerSeat);
    peer.pc.onicecandidate = null;
    peer.pc.ontrack = null;
    peer.pc.onconnectionstatechange = null;
    try {
      peer.pc.close();
    } catch (_error) {}
    peer.audio.srcObject = null;
    peer.audio.remove();
  }

  clearPeers() {
    for (const peerSeat of Array.from(this.peers.keys())) {
      this.removePeer(peerSeat);
    }
  }

  startLevelDetection() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !this.localStream) return;
    try {
      this.audioContext = new AudioContextClass();
      this.audioContext.resume().catch(() => {});
      const source = this.audioContext.createMediaStreamSource(this.localStream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 512;
      source.connect(this.analyser);
      const samples = new Uint8Array(this.analyser.fftSize);
      const tick = () => {
        if (!this.analyser || !this.localStream) return;
        this.analyser.getByteTimeDomainData(samples);
        let squared = 0;
        for (const sample of samples) {
          const normalized = (sample - 128) / 128;
          squared += normalized * normalized;
        }
        const rms = Math.sqrt(squared / samples.length);
        if (!this.muted && rms >= SPEAKING_RMS_THRESHOLD) {
          this.speakingFrames += 1;
          this.silentFrames = 0;
          if (this.speakingFrames >= SPEAKING_START_FRAMES) {
            this.setLocalSpeaking(true);
          }
        } else {
          this.silentFrames += 1;
          this.speakingFrames = 0;
          if (this.silentFrames >= SPEAKING_STOP_FRAMES) {
            this.setLocalSpeaking(false);
          }
        }
        this.levelFrame = window.requestAnimationFrame(tick);
      };
      this.levelFrame = window.requestAnimationFrame(tick);
    } catch (error) {
      console.warn("Voice level detection unavailable:", error);
    }
  }

  stopLevelDetection() {
    if (this.levelFrame) {
      window.cancelAnimationFrame(this.levelFrame);
      this.levelFrame = 0;
    }
    this.analyser = null;
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    this.speakingFrames = 0;
    this.silentFrames = 0;
  }

  setLocalSpeaking(speaking, notify = true) {
    const next = !!speaking && !this.muted;
    if (next === this.localSpeaking) return;
    this.localSpeaking = next;
    const ownSeat = this.joinContext?.seat;
    if (ownSeat) {
      if (next) this.speakingSeats.add(ownSeat);
      else this.speakingSeats.delete(ownSeat);
    }
    if (notify) this.sendVoiceState();
    this.emit();
  }

  stopLocalStream() {
    if (!this.localStream) return;
    this.localStream.getTracks().forEach((track) => track.stop());
    this.localStream = null;
  }
}

export function createVoiceChatController(options) {
  return new VoiceChatController(options);
}
