from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json
import logging
from io import BytesIO
import urllib, cStringIO
from PIL import Image
import base64

from utils import get_item, test_animal

hostName = "127.0.0.1"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):


    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        # self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))
        self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}))

    def do_POST(self):
        # content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        content_len = int(self.headers.get('Content-Length'))

        post_body = self.rfile.read(content_len)
        data = str(json.loads(post_body).get('data', {}))
        url = "new_image.png"
        image_64_decode = base64.decodestring(data)
        image_result = open(url, 'wb')  # create a writable image and write the decoding result
        image_result.write(image_64_decode)

        print (test_animal(url))

        response =json.dumps({'type': test_animal(url)})  # create response
        # logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n", str(self.headers), post_body.decode('utf-8'))
        self.send_response(200)  # create header
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()

        self.wfile.write(response)  # send response
        # self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}))



if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")