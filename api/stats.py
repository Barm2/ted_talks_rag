from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        # Your RAG configuration
        config = {
            "chunk_size": 1024,
            "overlap_ratio": 0.2,
            "top_k": 10
        }
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # Optional: for CORS
        self.end_headers()
        self.wfile.write(json.dumps(config).encode("utf-8"))
