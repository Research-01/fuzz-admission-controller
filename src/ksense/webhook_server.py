#!/usr/bin/env python3
import json
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fuzzy_controller import FuzzyController


def _json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class WebhookHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if self.path.startswith("/score"):
            report = self.server.controller.decide()
            _json_response(self, 200, report)
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        if self.path.startswith("/validate"):
            try:
                review = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                _json_response(self, 400, {"error": "invalid json"})
                return

            req = review.get("request", {}) or {}
            uid = req.get("uid", "")
            operation = (req.get("operation") or "").upper()
            kind = ((req.get("kind") or {}).get("kind") or "")

            # Always compute report for visibility/audit
            report = self.server.controller.decide()

            # Always allow Pod CREATE; scheduler keeps pods Pending on deny.
            always_allow = (operation == "CREATE" and kind == "Pod")

            allowed = True if always_allow else (report.get("decision") == "allow")

            msg = f"decision={report.get('decision')} score={float(report.get('score', 0.0)):.3f} level={report.get('level')}"
            if report.get("reason"):
                msg = f"{msg} reason={report['reason']}"

            if always_allow and report.get("decision") != "allow":
                msg = f"{msg} (CREATE override: allowed=true)"

            response = {
                "apiVersion": review.get("apiVersion", "admission.k8s.io/v1"),
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": allowed,
                    "status": {"message": msg},
                    "auditAnnotations": {
                        "fuzzy.score": f"{float(report.get('score', 0.0)):.3f}",
                        "fuzzy.level": str(report.get("level", "")),
                        "fuzzy.decision": str(report.get("decision", "")),
                        "fuzzy.operation": operation,
                        "fuzzy.kind": kind,
                        "fuzzy.create_override": "true" if always_allow else "false",
                    },
                },
            }

            if report.get("missing"):
                response["response"]["auditAnnotations"]["fuzzy.missing"] = ",".join(report["missing"])

            _json_response(self, 200, response)
            return

        if self.path.startswith("/score"):
            report = self.server.controller.decide()
            _json_response(self, 200, report)
            return

        _json_response(self, 404, {"error": "not found"})

    def log_message(self, format, *args):
        return


def run_server(host="0.0.0.0", port=8443, tls_cert=None, tls_key=None):
    controller = FuzzyController()

    # Important: this produces monitor.csv rows at 1 Hz and keeps CPU/PSI cached as true 1-second metrics.
    controller.start_monitoring(1.0)

    httpd = ThreadingHTTPServer((host, port), WebhookHandler)
    httpd.controller = controller

    if tls_cert and tls_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(tls_cert, tls_key)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        proto = "https"
    else:
        proto = "http"

    print(f"[fuzzy-webhook] listening on {proto}://{host}:{port}")
    httpd.serve_forever()
