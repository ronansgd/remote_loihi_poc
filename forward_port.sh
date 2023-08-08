echo "Enter your ncl-com username: "  
read USER_NAME  

MGMT_PORT=$(shuf -i 1024-65535 -n 1)
echo "Management port: $MGMT_PORT"

DATA_PORT=$(shuf -i 1024-65535 -n 1)
echo "Data port: $DATA_PORT"
ssh -t -L $MGMT_PORT:127.0.0.1:$MGMT_PORT -L $DATA_PORT:127.0.0.1:$DATA_PORT $USER_NAME@ncl-com.research.intel-research.net "export MGMT_PORT=$MGMT_PORT; export DATA_PORT=$DATA_PORT; bash -l"