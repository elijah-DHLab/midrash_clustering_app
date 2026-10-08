#!/bin/bash
# Add a user to the midrash explorer, or change their password. No sudo needed:
# nginx reads the file on every request, so the change is live at once.
#     bash set_password.sh <user>          (asks for the password, twice)
#     bash set_password.sh -d <user>       (removes the user)
set -euo pipefail
FILE=/home/adi/midrash/htpasswd
touch "$FILE"; chmod 644 "$FILE"
if [ "${1:-}" = "-d" ]; then
    grep -v "^$2:" "$FILE" > "$FILE.new" || true; mv "$FILE.new" "$FILE"; echo "removed $2"; exit 0
fi
USER_NAME="$1"
HASH=$(openssl passwd -6)          # prompts; SHA-512 crypt, which nginx reads
grep -v "^$USER_NAME:" "$FILE" > "$FILE.new" || true
echo "$USER_NAME:$HASH" >> "$FILE.new"; mv "$FILE.new" "$FILE"; chmod 644 "$FILE"
echo "set password for $USER_NAME"
