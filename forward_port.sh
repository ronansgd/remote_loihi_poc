echo "Enter your ncl-com username: "  
read USER_NAME  

LOIHI_PORT=$(shuf -i 1024-65535 -n 1)
echo "Generated port: $LOIHI_PORT"
ssh -t -L $LOIHI_PORT:127.0.0.1:$LOIHI_PORT $USER_NAME@ncl-com.research.intel-research.net "export LOIHI_PORT=$LOIHI_PORT; bash -l"