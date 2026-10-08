#!/bin/bash
# One-time root step for the midrash explorer. Run on the server as:
#     sudo bash /home/adi/midrash/webapp/deploy/install_root.sh
# Safe to run again. HAMATATE is not restarted; nginx is only reloaded, and
# only after its configuration has passed `nginx -t` - otherwise the original
# is put back and nothing changes.
set -euo pipefail
HERE=/home/adi/midrash/webapp/deploy
SITE=/etc/nginx/sites-available/hamatate
SNIP=/etc/nginx/snippets/midrash.conf
INCLUDE="    include $SNIP;"

install -m 644 "$HERE/midrash.service" /etc/systemd/system/midrash.service
mkdir -p /etc/nginx/snippets
install -m 644 "$HERE/nginx_midrash.conf" "$SNIP"

if ! grep -qF "$SNIP" "$SITE"; then
    cp -p "$SITE" "$SITE.before-midrash"
    # Before the catch-all `location / {` of the HAMATATE block.
    sed -i "0,/^\s*location \/ {/s||$INCLUDE\n\n    location / {|" "$SITE"
fi

if ! nginx -t 2>/tmp/nginx-test.log; then
    echo "nginx -t failed - restoring the original configuration:"; cat /tmp/nginx-test.log
    [ -f "$SITE.before-midrash" ] && cp -p "$SITE.before-midrash" "$SITE"
    rm -f "$SNIP"; exit 1
fi

systemctl daemon-reload
systemctl enable --now midrash.service
systemctl restart midrash.service
sleep 3
systemctl reload nginx
systemctl is-active midrash.service gunicorn.service nginx
echo "done: http://51.85.73.246/midrash/"
