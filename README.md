# Temporary working repo to develop a Socket-based client/server Process for Lava.

To run the server, use the following command:
`python -m cs_lava.demo_socket -s <address> <port>`

For example, to run the server process demo on your local machine, you could do:
`python -m cs_lava.demo_socket -s 127.0.0.1 12348`

To launch the client and connect to your running server, in a separate terminal or process:
`python -m cs_lava.demo_socket -c <address> <port>`

Address and port should match, and must be accessible from your client machine.

If your server is not running on a public or locally accessible machine, but you can access
the machine with SSH (e.g. Intel vLab), make sure you forward the correct port when
starting your SSH session:
`ssh <user>@<remote-machine> -L <address>:<port>:<address>:<port>
