read -p "Enter your VM name [default=ncl-com]: " VM_NAME
VM_NAME=${VM_NAME:-"ncl-com"}
echo "Selected VM: $VM_NAME"

read -p "Enter your $VM_NAME username: " USER_NAME
echo "Selected username: $USER_NAME"

export LOIHI_PORT=$(shuf -i 1024-65535 -n 1)
echo "Generated port: $LOIHI_PORT"

# NOTE: opens a  second terminal, where you can run the client with LOIHI_PORT
gnome-terminal

ssh -t -L $LOIHI_PORT:127.0.0.1:$LOIHI_PORT $USER_NAME@$VM_NAME.research.intel-research.net "export LOIHI_PORT=$LOIHI_PORT; bash -l"