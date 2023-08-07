# https://www.linuxjournal.com/content/using-named-pipes-fifos-bash

import subprocess


if __name__ == '__main__':
    sshProcess = subprocess.Popen(['ssh',
                                   '-tt',
                                   'ncl-com'],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  universal_newlines=True,
                                  bufsize=0)
    sshProcess.stdin.write("ls .\n")
    sshProcess.stdin.write("echo END\n")
    sshProcess.stdin.write("uptime\n")
    sshProcess.stdin.write("logout\n")
    sshProcess.stdin.close()
    
    
    for line in sshProcess.stdout:
        if line == "END\n":
            break
        print(line, end="")
    
    # to catch the lines up to logout
    for line in sshProcess.stdout:
        print(line, end="\n")
