# Public-room notices

In the administrator page, Site Settings > 公開部屋のお知らせ configures each public
room separately. Select a room, choose whether to display the notice, and select
通常のお知らせ or 1222のつぶやき. Save with 設定を保存.

Custom notices use the same limits as private-room notices: title 40 characters,
message 200 characters, optional HTTP(S) URL up to 2048 characters. An enabled
custom notice requires a message. The special mode preserves the existing text,
animation and click-count responses; it does not overwrite the saved custom text.
Newly deployed code defaults to the existing special notice until configured.

Settings are included in `__lobby__.public_room_ads` in the existing persistent
room-settings file. The legacy settings API accepts the new field too; omitting
it preserves current public notices. Failed persistent writes roll back changes.
Room state broadcasts carry only the active public notice; disabled notices and
inactive custom text are not sent for display. Private-room notices are unchanged.
