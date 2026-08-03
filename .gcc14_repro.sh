set -e
cmake -S . -B build/build-docker-gcc14 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DDFTRACER_UTILS_TESTS=ON -DDFTRACER_UTILS_ENABLE_MPI=OFF -DDFTRACER_UTILS_BUILD_PYTHON=OFF \
  -DDFTRACER_UTILS_COVERAGE=OFF -DCPM_SOURCE_CACHE=/work/.cpmsource >/dev/null
cmake --build build/build-docker-gcc14 -j 8 --target dftracer_view >/dev/null 2>&1
echo "BUILD_OK"
bin=/work/build/build-docker-gcc14/bin/dftracer_view
tmp=$(mktemp -d)
python3 - "$tmp" <<'PY'
import gzip, sys
d=sys.argv[1]; io=["read","write","open","close","pread","pwrite","fread","fwrite"]; cats=["POSIX"]*6+["STDIO"]*2
out=[]
for i in range(1,21):
    name=io[i%8]; cat=cats[i%8]
    line=('{"name":"%s","cat":"%s","pid":%d,"tid":%d,"ts":%d,"dur":%d,"ph":"X","args":{"ret":%d,"hhash":"abc123","data":"'%(name,cat,1000+i%4,2000+i%8,1000000+i*1000,(i*123%10000),1024*i))
    pad=max(0,1024-len(line)-3); line+="x"*pad+'"}}\n'; out.append(line)
with gzip.open(d+"/test_data.pfw.gz","wt") as f:
    f.write("[\n"); f.writelines(out); f.write("]\n")
PY
echo "=== gcc14 aggregate ==="
$bin --directory $tmp --group-by name --agg mean:dur 2>&1 | grep -v "INFO\|Auto-"
