# Member accounts: initial release

This release adds manually issued member accounts, first-login password change,
a My Page tab in the existing settings dialogs, and administrator resets.
It does NOT yet enable all stamps or personal kifu storage in public rooms.
Existing private-room management and shared kifu permissions are unchanged.

## Production configuration

- Set `GOITA_PERSISTENT_DATA_DIR=/var/data` to the existing Render disk mount.
  Accounts and hashed sessions live in `/var/data/goita-members.sqlite3`.
  `GOITA_MEMBER_DB_PATH` can override this path. Do not put the database under
  `frontend`, which is publicly served. On Render, member management refuses
  to operate without a configured persistent path.
- Set `LOBBY_ADMIN_PASSWORD` to a unique, strong administrator password. The
  known default `admin-lobby` is intentionally rejected for member management.
- HTTPS is required in production. Member cookies are HttpOnly, host-only,
  SameSite=Strict, and Secure on HTTPS (always Secure on Render). Uvicorn's
  trusted proxy configuration must report the external HTTPS scheme correctly.
  Use the same canonical hostname for login and subsequent use: the custom
  domain and the onrender.com domain have separate cookies.
- No new Python dependency or automatic billing integration is required.

## Administrator workflow

1. Open `/admin/`, authenticate, and select `会員管理`.
2. After confirming payment, issue a member ID (4-32 ASCII letters, digits,
   hyphen or underscore, starting with a letter/digit; case-insensitive).
3. Set paid access and its expiry date, or leave expiry blank for no expiry.
   Dates include the entire specified day in Japan Standard Time.
4. Privately deliver the generated temporary password to the verified member.
   It is shown only in that response and expires after 24 hours. Closing or
   leaving the tab clears the displayed credential. A lost credential must be
   reset; it cannot be retrieved.
5. The member selects `ログイン`, then must replace the temporary password with
   a 15-128 character password. The initial session can only change passwords
   or log out; it cannot access `/api/member/me` or paid authorization.

Remembered sessions last 30 days, with up to 10 devices per member. Changing
the password invalidates all existing sessions and issues one new session for
the current device. A reset invalidates all sessions and the old password.
Account suspension invalidates all sessions; reenabling does not restore them.
Paid-access expiry/disable is separate from account suspension and does not
prevent logging in. This is intentional for future access to saved records.
The site-administrator login is limited to five failed attempts per 15-minute
process window because it can now issue and reset member credentials.

Before a reset, verify the requester against your payment/support records;
knowing the member ID alone is not proof of identity. A reset does not lift
an account suspension or change paid access. No email address is required.

## Security and data boundary

Passwords use salted PBKDF2-HMAC-SHA256 with 600,000 iterations, following the
[OWASP password-storage guidance](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html#pbkdf2).
Only hashes of random session tokens are persisted. All member/admin-member
APIs require a custom header and reject foreign origins, including reads;
credentials and validation responses are not cached. Credentials are never
written to browser storage, analytics, game objects, or logs by these modules.

Login and password-change attempts share per-member throttling (5 attempts
per 15 minutes, cleared on successful verification) and a global 10/minute
budget. Short-lived HMAC-keyed counters survive process restarts and are
purged as new attempts arrive. They contain no raw IP addresses. These limits
are deliberately conservative for a small manually managed membership.

Store the database outside version control and public static directories.
Use SQLite's backup API for backups of a running service, and protect backups
as credentials. Restore the member database separately from analytics. There
is no delete-account UI, payment provider integration, or email recovery yet.

## Verification

Run `python -m pytest tests/test_member_accounts.py` in the project environment.
Exercise desktop/mobile issuance, initial change, reopen, logout, reset, and
suspension using a temporary local database before issuing real credentials.
