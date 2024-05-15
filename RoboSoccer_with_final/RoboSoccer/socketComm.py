import sys
import socket
import selectors
import types
from prediction import AgentPrediction


sel = selectors.DefaultSelector()


clients = {}
Agents={}

host, port = '172.17.7.69', 8080


def accept_wrapper(sock):
    conn, addr = sock.accept() 
    print(f"Accepted connection from {addr}")
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)
    clients[addr] = conn

    

def service_connection(key, mask):
    sock = key.fileobj
    data = key.data
    

    if mask & selectors.EVENT_WRITE:
        if data.outb:
            print(f"Echoing {data.outb!r} to {data.addr}")
            sent = sock.send(data.outb) 
            data.outb = data.outb[sent:]

    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)
        if recv_data:
            recv_str = recv_data.decode().strip()
           
            if recv_str.startswith("ID:"):
                try:
                    conn_id = int(recv_str.split(":")[1])
                    Agents[conn_id]= AgentPrediction(conn_id,clients[key.data.addr])
                    print("Agent connected with ID:",conn_id)
                    print(Agents)

                except (ValueError, IndexError):
                    print("Invalid connection ID format received from client.")
        else:
            print(f"Connection closed by {data.addr}")
            sel.unregister(sock)
            # sock.close()
            clients.pop(data.addr)
            key = {Agents[i].id for i in Agents if Agents[i].conn==sock}
            if(key!=None):
                Agents[key].close()
                del Agents[key]
          
def main():
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.bind((host, port))
    lsock.listen()
    print(f"Listening on {(host, port)}")
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)

    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    service_connection(key, mask)
            

    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
    finally:
        sel.close()
        lsock.close()

if __name__=="__main__":
    main()