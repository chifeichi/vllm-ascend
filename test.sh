SO="$(ldd "$(python - <<'PY'
import mooncake.engine
print(mooncake.engine.__file__)
PY
)" | awk '/ascend_transport\.so/{print $3}')"

echo "$SO"
nm -D "$SO" | c++filt | grep 'AscendDirectTransport::disconnectAllPeers'