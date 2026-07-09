# Expose The API Through A Tunnel

Use this when the FastAPI backend runs on a separate machine where you do not
have root access, and the frontend is hosted somewhere else.

## Backend Command

Run the API on the backend machine:

```bash
CORS_ALLOW_ORIGINS=https://your-frontend.example.com \
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

For local testing, allow both the hosted frontend and localhost:

```bash
CORS_ALLOW_ORIGINS=https://your-frontend.example.com,http://localhost:3000,http://localhost:5173 \
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Keep `--host 127.0.0.1` when using a tunnel. The API only needs to be reachable
by the tunnel process on the same machine.

## Tunnel Options Without Root

### Option A: Cloudflare Tunnel

Works well on machines without root if you can download a user-local binary.

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

The command prints a public HTTPS URL. Use that URL as the frontend API base.

### Option B: ngrok

```bash
ngrok http 8000
```

Use the printed HTTPS forwarding URL as the frontend API base.

### Option C: SSH Reverse Tunnel

Use this if you have SSH access to a public machine:

```bash
ssh -N -R 8000:127.0.0.1:8000 user@public-machine.example.com
```

Then expose `http://public-machine.example.com:8000` or put HTTPS/proxying in
front of that public machine.

## Point The Frontend At The Tunnel

The HTML frontend now supports a configurable API base URL.

Open it with:

```text
https://your-frontend.example.com/?api=https://your-tunnel-url.example.com
```

The `api` query parameter is saved in browser `localStorage`, so subsequent
loads of the same frontend can omit it.

You can also set it before loading the HTML:

```html
<script>
  window.API_BASE_URL = "https://your-tunnel-url.example.com";
</script>
```

All API calls then go to:

```text
https://your-tunnel-url.example.com/api/upload
https://your-tunnel-url.example.com/api/stream/{run_id}
https://your-tunnel-url.example.com/api/query/{run_id}
https://your-tunnel-url.example.com/api/approve/{run_id}
https://your-tunnel-url.example.com/api/health
```

## Quick Health Check

From your frontend machine or browser:

```bash
curl https://your-tunnel-url.example.com/api/health
```

If this works but the browser frontend fails, check `CORS_ALLOW_ORIGINS`.
It must contain the exact frontend origin, including scheme and port.

## Notes

- Server-Sent Events (`/api/stream/{run_id}`) must be supported by the tunnel.
  Cloudflare Tunnel and ngrok both support this for normal HTTP forwarding.
- Do not use `CORS_ALLOW_ORIGINS=*` unless this is a throwaway demo. The upload
  endpoint accepts PDFs and can trigger expensive local model runs.
- The backend run state is still process-local. Upload, stream, and approval for
  a given run must hit the same backend process.
