#!/bin/bash

docker run --name judge --network host -v /home/nccsoft/ncc-erp-contest/dmoj/problems:/problems --cap-add=SYS_PTRACE -d --restart=always dmoj/judge-tier2:latest run -p "9999" -c /problems/judge.yml "127.0.0.1" "ncc-tier2" "4fg/NMnIHi5NWGILMnAi7Qt7iSVTnDednfBx2A5GRbDEyTiH/+vHFKqg+taNLNb53VYbsj+Cno0cG8xi/7LqL3VKUhqyiOPpXDyN"