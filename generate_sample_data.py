# ─────────────────────────────────────────────
# generate_sample_data.py
# Generates synthetic IT tickets CSV + sample SOP PDFs
# Run this FIRST if you don't have real data
# ─────────────────────────────────────────────

import os
import csv
import json
from pathlib import Path

def generate_tickets_csv(path="data/tickets.csv", count=30):
    """Generate synthetic IT support tickets."""
    os.makedirs("data", exist_ok=True)

    tickets = [
        {"id": 1, "title": "Cannot SSH into production server",
         "description": "Getting permission denied error when trying to SSH",
         "resolution": "SSH key not added to authorized_keys. Run: ssh-copy-id user@server",
         "category": "Access"},
        {"id": 2, "title": "Docker container exits with code 137",
         "description": "Container keeps restarting automatically",
         "resolution": "OOM error. Increase memory limit in docker-compose.yml using mem_limit: 512m",
         "category": "Infrastructure"},
        {"id": 3, "title": "Database connection refused on port 5432",
         "description": "Application cannot connect to PostgreSQL database",
         "resolution": "Start PostgreSQL: sudo systemctl start postgresql. Check firewall rules for port 5432",
         "category": "Database"},
        {"id": 4, "title": "Kubernetes pod stuck in CrashLoopBackOff",
         "description": "Pod keeps crashing and restarting every few minutes",
         "resolution": "Check logs with: kubectl logs <pod-name>. Usually caused by missing env variables or wrong image tag",
         "category": "Kubernetes"},
        {"id": 5, "title": "Jenkins pipeline failing at build stage",
         "description": "Build fails with npm install error",
         "resolution": "Clear npm cache: npm cache clean --force. Delete node_modules and reinstall",
         "category": "CI/CD"},
        {"id": 6, "title": "Git merge conflict in main branch",
         "description": "Cannot merge feature branch due to conflicts",
         "resolution": "Run git mergetool or manually edit conflict markers (<<<<, ====, >>>>) in files. Then git add and commit",
         "category": "Version Control"},
        {"id": 7, "title": "API returning 502 Bad Gateway",
         "description": "Load balancer showing 502 for all requests",
         "resolution": "Backend service is down. Check: kubectl get pods. Restart deployment: kubectl rollout restart deployment/api",
         "category": "API"},
        {"id": 8, "title": "Python package install fails with permission error",
         "description": "pip install returns permission denied",
         "resolution": "Use virtual environment: python -m venv venv && source venv/bin/activate. Then pip install without sudo",
         "category": "Python"},
        {"id": 9, "title": "Kafka consumer not receiving messages",
         "description": "Consumer group is connected but messages not being consumed",
         "resolution": "Check consumer group offset: kafka-consumer-groups.sh --describe. Reset offset if needed with --reset-offsets flag",
         "category": "Kafka"},
        {"id": 10, "title": "SSL certificate expired on production domain",
         "description": "Browser showing certificate error for our domain",
         "resolution": "Renew cert with certbot: sudo certbot renew. Then reload nginx: sudo systemctl reload nginx",
         "category": "Security"},
        {"id": 11, "title": "High CPU usage on application server",
         "description": "Server CPU consistently above 90%",
         "resolution": "Use top/htop to identify process. Check for infinite loops or memory leaks. Consider horizontal scaling",
         "category": "Performance"},
        {"id": 12, "title": "Cannot access internal wiki page",
         "description": "Getting 403 Forbidden on Confluence pages",
         "resolution": "Request access from space admin. Go to Space Settings > Permissions and add your user",
         "category": "Access"},
        {"id": 13, "title": "Redis cache not working in production",
         "description": "Application falling back to database for every request",
         "resolution": "Redis service stopped. Run: sudo systemctl start redis. Verify with: redis-cli ping",
         "category": "Cache"},
        {"id": 14, "title": "FastAPI returning 422 Unprocessable Entity",
         "description": "POST request failing with validation error",
         "resolution": "Request body does not match Pydantic model schema. Check field names and types match the model definition",
         "category": "API"},
        {"id": 15, "title": "New engineer cannot clone repository",
         "description": "Git clone failing with authentication error",
         "resolution": "Add SSH public key to GitLab profile. Generate key: ssh-keygen -t ed25519. Add ~/.ssh/id_ed25519.pub to GitLab",
         "category": "Onboarding"},
    ]

    # Pad to requested count
    while len(tickets) < count:
        t = tickets[len(tickets) % len(tickets[:15])].copy()
        t["id"] = len(tickets) + 1
        t["title"] = t["title"] + f" (variant {len(tickets)})"
        tickets.append(t)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id","title","description","resolution","category"])
        writer.writeheader()
        writer.writerows(tickets[:count])

    print(f"✓ Generated {count} tickets → {path}")


def generate_sample_docs(docs_dir="data/documents"):
    """Generate sample SOP text files."""
    os.makedirs(docs_dir, exist_ok=True)

    docs = {
        "onboarding_sop.txt": """
EMPLOYEE ONBOARDING SOP — IT Services Company

1. SYSTEM ACCESS SETUP
   - Submit IT access request form on HR portal
   - Request VPN credentials from IT helpdesk (helpdesk@company.com)
   - VPN client: Cisco AnyConnect, server: vpn.company.com
   - Setup two-factor authentication (Google Authenticator)

2. DEVELOPMENT ENVIRONMENT
   - Install required tools: Git, Docker, kubectl, Python 3.11, Node.js 20
   - Clone main repository: git clone git@gitlab.company.com:platform/main.git
   - Run setup script: chmod +x setup.sh && ./setup.sh
   - Configure Git user: git config --global user.email "your@company.com"

3. ACCOUNTS & ACCESS
   - GitLab: Request maintainer access from team lead
   - Jira: Auto-provisioned with company email
   - Confluence: Request space access from project manager
   - AWS Console: Submit cloud access request with justification

4. SECURITY REQUIREMENTS
   - Complete mandatory security training on LMS within first week
   - Never commit credentials or API keys to repositories
   - Enable full-disk encryption on laptop
   - Lock screen when away from desk (Win+L / Cmd+Ctrl+Q)

5. FIRST WEEK CHECKLIST
   - Day 1: System setup + meet team
   - Day 2: Review architecture documentation on Confluence
   - Day 3: Complete security training module
   - Day 4: Shadow senior engineer on current sprint tasks
   - Day 5: Set up 1:1 with team lead (recurring weekly)
""",

        "deployment_sop.txt": """
DEPLOYMENT PROCEDURE SOP — Production Releases

PRE-DEPLOYMENT CHECKLIST
- All tests passing in CI/CD pipeline
- Code review approved by minimum 2 reviewers
- QA sign-off on staging environment
- Database migrations tested on staging
- Rollback plan documented

DEPLOYMENT STEPS
1. Create deployment ticket in Jira with change description
2. Notify stakeholders via #deployments Slack channel
3. Take database backup: pg_dump -U admin production_db > backup_$(date +%Y%m%d).sql
4. Deploy to production: kubectl apply -f k8s/production/
5. Monitor deployment: kubectl rollout status deployment/api
6. Run smoke tests against production endpoints
7. Monitor error rates in Grafana for 15 minutes post-deploy
8. Update deployment ticket with completion timestamp

ROLLBACK PROCEDURE
If issues detected within 1 hour:
1. kubectl rollout undo deployment/api
2. Restore database if schema changes: psql -U admin production_db < backup_file.sql
3. Notify team in #incidents channel
4. Create post-mortem ticket

DEPLOYMENT WINDOWS
- Regular releases: Tuesday/Thursday 10:00-12:00 IST
- Hotfixes: Any time with on-call engineer approval
- Freeze period: Last 5 days of each quarter
""",

        "incident_response_sop.txt": """
INCIDENT RESPONSE SOP

SEVERITY LEVELS
- P1 (Critical): Production down, data loss risk → Response: 15 minutes
- P2 (High): Major feature unavailable → Response: 1 hour
- P3 (Medium): Performance degraded → Response: 4 hours
- P4 (Low): Minor issue, workaround available → Response: Next business day

INCIDENT RESPONSE STEPS
1. DETECT: Alert received via PagerDuty or Grafana
2. ACKNOWLEDGE: On-call engineer acknowledges within SLA window
3. COMMUNICATE: Post in #incidents Slack channel:
   "🚨 P[N] Incident: [brief description] | Status: Investigating | IC: [name]"
4. INVESTIGATE: Check logs in Kibana, metrics in Grafana, traces in Jaeger
5. CONTAIN: Apply immediate fix or rollback to stop bleeding
6. RESOLVE: Implement proper fix, verify metrics return to normal
7. COMMUNICATE: Post resolution message in #incidents
8. POST-MORTEM: Schedule blameless post-mortem within 48 hours for P1/P2

USEFUL COMMANDS
- Check pod logs: kubectl logs <pod-name> --tail=100
- Describe failing pod: kubectl describe pod <pod-name>
- Check service endpoints: kubectl get endpoints
- Restart deployment: kubectl rollout restart deployment/<name>
- Check resource usage: kubectl top pods
"""
    }

    for filename, content in docs.items():
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(content.strip())
        print(f"✓ Generated {filepath}")


def generate_eval_dataset(path="data/eval_dataset.json"):
    """Generate evaluation dataset."""
    dataset = [
        {
            "question": "How do I restart a Kubernetes pod?",
            "relevant_sources": ["kubernetes-pods"],
            "answer_keywords": ["kubectl", "delete", "pod"]
        },
        {
            "question": "How to set up a Kafka consumer in Python?",
            "relevant_sources": ["kafka-consumer"],
            "answer_keywords": ["confluent-kafka", "subscribe", "poll"]
        },
        {
            "question": "What is Docker?",
            "relevant_sources": ["docker-intro"],
            "answer_keywords": ["container", "application", "image"]
        },
        {
            "question": "How to create Python virtual environment?",
            "relevant_sources": ["python-venv"],
            "answer_keywords": ["venv", "activate", "pip"]
        },
        {
            "question": "New engineer onboarding steps",
            "relevant_sources": ["onboarding_sop.txt"],
            "answer_keywords": ["VPN", "Git", "Docker", "security"]
        },
        {
            "question": "Docker container exits with code 137",
            "relevant_sources": ["tickets.csv"],
            "answer_keywords": ["memory", "OOM", "mem_limit"]
        },
        {
            "question": "Database connection refused port 5432",
            "relevant_sources": ["tickets.csv"],
            "answer_keywords": ["PostgreSQL", "systemctl", "firewall"]
        },
        {
            "question": "How to run FastAPI application?",
            "relevant_sources": ["fastapi-setup"],
            "answer_keywords": ["uvicorn", "main:app", "pip"]
        },
    ]

    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✓ Generated eval dataset → {path}")


if __name__ == "__main__":
    print("🔧 Generating sample data...\n")
    generate_tickets_csv()
    generate_sample_docs()
    generate_eval_dataset()
    print("\n✅ Sample data ready! Now run: python ingest.py")
