#!/bin/bash

docker run \
--name judge \
--network host \
-v $(cd ../../ ; pwd)/dmoj/problems:/problems \
--cap-add=SYS_PTRACE \
-d \
--restart=always \
dmoj/judge-tier2:latest \
run -p "9999" -e="V8JS,C,C11,CPP03,CPP11,CPP14,CPP17,CPP20,DART,GO,JAVA8,JAVA9,JAVA10,JAVA11,JAVA15,JAVA17,KOTLIN,LUA,MONOCS,PAS,PHP,PY2,PY3,SWIFT,V8JS" -c /problems/judge.yml \
"127.0.0.1" "ncc-tier2" "4fg/NMnIHi5NWGILMnAi7Qt7iSVTnDednfBx2A5GRbDEyTiH/+vHFKqg+taNLNb53VYbsj+Cno0cG8xi/7LqL3VKUhqyiOPpXDyN"
