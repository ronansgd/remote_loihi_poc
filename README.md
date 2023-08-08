# remote_loihi_poc

## Install the project

- Create a python virtual environment `env` with Lava
- Install the project lib in edition mode

```
$ pip install -e src/
```

## Forward port via SSH

First, we generate a random port id in the range of non-privileged ports:
```
$ LOIHI_PORT=$(shuf -i 1024-65535 -n 1)
$ echo $LOIHI_PORT
```
Then, we start a ssh session forwarding the generated port id:
```
$ ssh -t -L $LOIHI_PORT:127.0.0.1:$LOIHI_PORT your_username@ncl-com.research.intel-research.net ncl-com "export LOIHI_PORT=$LOIHI_PORT; bash -l"
```
Side notes: 
- don't forget to set your username.
- the `-t` flag and the final command `bash -l` allow to keep the session alive after the `LOIHI_PORT` variable is forwarded.
- the `-L` flag allows to forward the local port of id `LOIHI_PORT` to the matching remote port.

Finally, we can test a basic client-server communication over the forwarded port.
In the ssh bash, run:
```
(env)$ python demos/numpy_protocol/numpy_server.py --port "$LOIHI_PORT"
```
Then, in a local bash, run (note that you have to manually copy the generated port value):
```
(env)$ LOIHI_PORT=XXXXX
(env)$ python demos/numpy_protocol/numpy_client.py --port "$LOIHI_PORT"
```

You should see the two sides exchanging basic messages!

*NOTE:  the workflow could be improved in the future using subprocess.Popen*