import argparse
import json
import sys
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from tcyz.runner import TCYZRunner

class TCYZRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Always send CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/model':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Return model metadata
            model_info = {
                'name': runner_instance.metadata.get('model_name', 'TCYZ Model'),
                'arch': {
                    'dim': runner_instance.args.dim,
                    'n_layers': runner_instance.args.n_layers,
                    'n_heads': runner_instance.args.n_heads,
                    'n_kv_heads': runner_instance.args.n_kv_heads,
                    'vocab_size': runner_instance.args.vocab_size,
                    'hidden_dim': runner_instance.args.hidden_dim,
                    'max_seq_len': runner_instance.args.max_seq_len,
                    'tokenizer_name': runner_instance.metadata.get('tokenizer_name', 'gpt2'),
                },
                'total_mb': len(runner_instance.tensors) # Estimation placeholder or real size if available
            }
            self.wfile.write(json.dumps(model_info).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == '/api/generate':
            # Read content length
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                params = json.loads(post_data.decode('utf-8'))
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON")
                return

            prompt = params.get('prompt', '')
            temperature = float(params.get('temperature', 0.7))
            top_k = int(params.get('top_k', 40))
            top_p = float(params.get('top_p', 0.9))
            max_tokens = int(params.get('max_tokens', 128))

            if not prompt:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing 'prompt'")
                return

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            print(f"Generating for prompt: {prompt[:50]}...")

            def stream_callback(token):
                try:
                    # Format as Server-Sent Event
                    data = json.dumps({'token': token})
                    self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except Exception as e:
                    # Client disconnected or socket closed
                    raise GeneratorExit("Client disconnected")

            try:
                runner_instance.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stream_callback=stream_callback
                )
                # Send end token event
                self.wfile.write(b"event: done\ndata: [DONE]\n\n")
                self.wfile.flush()
            except GeneratorExit:
                print("Client closed the connection during generation.")
            except Exception as e:
                print(f"Error during generation: {e}")
                try:
                    err_msg = json.dumps({'error': str(e)})
                    self.wfile.write(f"data: {err_msg}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except Exception:
                    pass
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

runner_instance = None

def main():
    global runner_instance
    parser = argparse.ArgumentParser(description="Run an API server for TCYZ models")
    parser.add_argument("--model", type=str, required=True, help="Path to .tcyz model file")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    print(f"Initializing TCYZ model on {args.device}...")
    try:
        runner_instance = TCYZRunner(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, TCYZRequestHandler)
    print(f"TCYZ Server running at http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/api/model")
    print(f"  POST http://{args.host}:{args.port}/api/generate")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TCYZ Server...")
        httpd.server_close()

if __name__ == '__main__':
    main()
