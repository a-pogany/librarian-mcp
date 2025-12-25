# Librarian MCP - Enterprise Enhancements Specification

**Version**: 3.0.0-enterprise (Planning)
**Created**: 2025-12-25
**Author**: Claude Code / APY
**Status**: 📋 PLANNING

---

## Executive Summary

This document specifies the enterprise transformation of Librarian MCP from a single-user documentation search tool to a multi-tenant enterprise platform. The transformation addresses critical requirements:

1. **Multi-User Support**: Isolated storage and search for multiple users
2. **Authentication & Authorization**: Secure credential management with role-based access
3. **Background Indexing**: Non-blocking indexing that doesn't disrupt the MCP server
4. **Real-time Status Tracking**: Per-user indexing progress visible in UI
5. **Enterprise Features**: Audit logging, admin dashboard, quota management

### Key Metrics

| Metric | Current (v2.2.0) | Target (v3.0.0) |
|--------|------------------|-----------------|
| Concurrent Users | 1 | 100+ |
| Storage per User | N/A | 5 GB |
| Indexing Mode | Blocking | Background |
| Server Availability | 0% during indexing | 99.9% |
| Data Isolation | None | Complete |

---

## Architecture Overview

### Current Architecture (v2.2.0)
```
┌─────────────────────────────────────────────────────────────────┐
│                      Single-User System                          │
├─────────────────────────────────────────────────────────────────┤
│  Browser ──► Agent Layer (4010) ──► MCP Backend (3001)          │
│                                            │                     │
│                                     docs/ (shared)               │
│                                     chromadb/ (single)           │
└─────────────────────────────────────────────────────────────────┘
```

### Enterprise Architecture (v3.0.0)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Multi-Tenant Enterprise System                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐   ┌─────────────────┐   ┌─────────────────────────────┐  │
│  │ Browser  │──►│  Agent Layer    │──►│     MCP Backend             │  │
│  │ (8080)   │   │  (4010)         │   │     (3001)                  │  │
│  │          │   │  + Auth Proxy   │   │     + Multi-tenant API      │  │
│  └──────────┘   └─────────────────┘   └─────────────────────────────┘  │
│                          │                         │                     │
│                          ▼                         ▼                     │
│                 ┌─────────────────┐    ┌─────────────────────────────┐  │
│                 │   PostgreSQL    │    │      Redis Job Queue        │  │
│                 │   - Users       │    │      - Index Jobs           │  │
│                 │   - Sessions    │    │      - Progress Tracking    │  │
│                 │   - Audit Logs  │    └─────────────────────────────┘  │
│                 └─────────────────┘                 │                    │
│                                                     ▼                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Indexing Worker Pool                            │  │
│  │   Worker 1 ◄─── Job Queue ───► Worker 2 ◄───► Worker N            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    │                                     │
│                                    ▼                                     │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Per-User Storage                               │  │
│  │  /data/users/{user_id}/                                           │  │
│  │      ├── documents/     (uploaded docs, max 5GB)                  │  │
│  │      ├── emails/        (uploaded .eml files)                     │  │
│  │      ├── chromadb/      (isolated vector index)                   │  │
│  │      └── metadata.json  (quota usage, settings)                   │  │
│  │                                                                    │  │
│  │  /data/shared/          (optional company-wide docs)              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Agent Layer** | Authentication, request routing, user context injection |
| **MCP Backend** | Multi-tenant search, document retrieval, tenant isolation |
| **PostgreSQL** | User accounts, sessions, audit logs, job metadata |
| **Redis** | Job queue, real-time progress, session cache |
| **Indexing Workers** | Background document processing, vector generation |
| **Per-User Storage** | Isolated file storage with quota enforcement |

---

## Phase E1: Authentication & User Management

> **Priority**: 🔴 CRITICAL
> **Estimated Effort**: 2-3 days
> **Dependencies**: None (foundational)

### E1-US-001: User Registration

**Story**: As a new user, I want to register with email and password, so that I can create my private document space.

**Acceptance Criteria**:
- [ ] Registration form with email, password, confirm password, display name
- [ ] Email validation (format check + uniqueness in database)
- [ ] Password requirements enforced:
  - Minimum 8 characters
  - At least 1 uppercase letter
  - At least 1 number
  - At least 1 special character
- [ ] On success: redirect to login page with "Registration successful" message
- [ ] On error: display inline validation errors (don't clear form)
- [ ] Create user's storage directory structure on registration

**Technical Notes**:
```
Files to create/modify:
  backend/core/auth/__init__.py     # New auth module
  backend/core/auth/models.py       # User SQLAlchemy model
  backend/core/auth/service.py      # Auth business logic
  backend/core/auth/routes.py       # FastAPI routes
  agent_layer/src/routes/auth.js    # Proxy auth endpoints
  frontend/librarian-ui/register.html  # Registration page
  frontend/librarian-ui/js/auth.js     # Auth client logic

Database Schema (User):
  CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',  -- 'user' | 'admin'
    storage_quota_bytes BIGINT DEFAULT 5368709120,  -- 5GB
    storage_used_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
  );

API Endpoint:
  POST /api/auth/register
  Body: { email, password, display_name }
  Response: { success: true, message: "Registration successful" }
           OR { success: false, errors: [...] }

Password Hashing:
  - Use bcrypt with work factor 12
  - pip install bcrypt passlib[bcrypt]
```

**Complexity**: M
**Dependencies**: None
**Status**: [ ] Not Started

---

### E1-US-002: User Login

**Story**: As a registered user, I want to log in with my credentials, so that I can access my documents.

**Acceptance Criteria**:
- [ ] Login form with email and password fields
- [ ] "Remember me" checkbox (extends session to 30 days)
- [ ] On success:
  - Generate JWT access token (15 min expiry)
  - Generate refresh token (7 days or 30 days if "remember me")
  - Store tokens in httpOnly cookies
  - Redirect to main search page
- [ ] On failure: display "Invalid credentials" (don't reveal which field is wrong)
- [ ] Rate limiting: 5 failed attempts → 15 minute lockout
- [ ] Update last_login_at timestamp on successful login

**Technical Notes**:
```
Files to modify:
  backend/core/auth/service.py      # Login logic
  backend/core/auth/routes.py       # Login endpoint
  backend/core/auth/jwt.py          # JWT generation/validation
  agent_layer/src/middleware/auth.js  # JWT validation middleware
  frontend/librarian-ui/login.html    # Login page
  frontend/librarian-ui/js/auth.js    # Token handling

JWT Payload:
  {
    "sub": "user-uuid",
    "email": "user@example.com",
    "role": "user",
    "iat": 1703500000,
    "exp": 1703500900  // 15 minutes
  }

API Endpoints:
  POST /api/auth/login
  Body: { email, password, remember_me }
  Response: Sets httpOnly cookies + { success: true, user: {...} }

  POST /api/auth/refresh
  Cookies: refresh_token
  Response: New access_token cookie

Security:
  - httpOnly cookies prevent XSS token theft
  - SameSite=Strict for CSRF protection
  - Secure flag in production (HTTPS only)
```

**Complexity**: M
**Dependencies**: E1-US-001
**Status**: [ ] Not Started

---

### E1-US-003: Password Reset

**Story**: As a user who forgot my password, I want to reset it via email, so that I can regain access to my account.

**Acceptance Criteria**:
- [ ] "Forgot password?" link on login page
- [ ] Password reset request form (email only)
- [ ] Always show "If email exists, reset link sent" (prevent enumeration)
- [ ] Reset token valid for 1 hour
- [ ] Reset link leads to new password form
- [ ] New password must meet same requirements as registration
- [ ] Invalidate all existing sessions after password reset

**Technical Notes**:
```
Files to create/modify:
  backend/core/auth/password_reset.py  # Reset token logic
  backend/core/email/service.py        # Email sending (SMTP or SendGrid)
  frontend/librarian-ui/forgot-password.html
  frontend/librarian-ui/reset-password.html

Database Schema:
  CREATE TABLE password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
  );

API Endpoints:
  POST /api/auth/forgot-password
  Body: { email }

  POST /api/auth/reset-password
  Body: { token, new_password }

Email Template:
  Subject: "Librarian - Password Reset Request"
  Body: Link with token (valid 1 hour)
```

**Complexity**: M
**Dependencies**: E1-US-002
**Status**: [ ] Not Started

---

### E1-US-004: JWT Token Management

**Story**: As the system, I want to manage JWT tokens securely, so that user sessions are protected and performant.

**Acceptance Criteria**:
- [ ] Access tokens expire after 15 minutes
- [ ] Refresh tokens expire after 7 days (or 30 if "remember me")
- [ ] Automatic token refresh when access token expires
- [ ] Token blacklist for logout/password reset
- [ ] Middleware validates tokens on all protected endpoints

**Technical Notes**:
```
Files to create/modify:
  backend/core/auth/jwt.py           # JWT encode/decode/validate
  backend/core/auth/middleware.py    # FastAPI dependency
  agent_layer/src/middleware/auth.js # Express middleware

Token Blacklist (Redis):
  Key: "blacklist:{jti}"
  Value: "1"
  TTL: remaining token lifetime

Dependencies:
  pip install python-jose[cryptography]
  npm install jsonwebtoken

Environment Variables:
  JWT_SECRET_KEY=<random-256-bit-key>
  JWT_ALGORITHM=HS256
  JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
  JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

**Complexity**: M
**Dependencies**: E1-US-002
**Status**: [ ] Not Started

---

### E1-US-005: User Profile Management

**Story**: As a logged-in user, I want to view and update my profile, so that I can manage my account settings.

**Acceptance Criteria**:
- [ ] Profile page shows: display name, email, storage usage, account created date
- [ ] Edit display name (live update)
- [ ] Change password (requires current password)
- [ ] Storage usage visualization (progress bar showing X/5GB)
- [ ] Cannot change email (security - would need verification flow)

**Technical Notes**:
```
Files to create/modify:
  backend/core/auth/routes.py          # Profile endpoints
  frontend/librarian-ui/profile.html   # Profile page
  frontend/librarian-ui/js/profile.js  # Profile client logic

API Endpoints:
  GET /api/user/profile
  Response: { id, email, display_name, storage_used, storage_quota, created_at }

  PATCH /api/user/profile
  Body: { display_name }

  POST /api/user/change-password
  Body: { current_password, new_password }
```

**Complexity**: S
**Dependencies**: E1-US-002
**Status**: [ ] Not Started

---

### E1-US-006: Session Logout

**Story**: As a logged-in user, I want to log out, so that my session is securely terminated.

**Acceptance Criteria**:
- [ ] Logout button in header (visible on all pages when logged in)
- [ ] On click: clear all auth cookies
- [ ] Add current tokens to blacklist
- [ ] Redirect to login page
- [ ] "Logout all devices" option in profile (invalidates all refresh tokens)

**Technical Notes**:
```
API Endpoints:
  POST /api/auth/logout
  Response: Clears cookies, returns { success: true }

  POST /api/auth/logout-all
  Response: Invalidates all user's refresh tokens

Redis Operations:
  # Add to blacklist
  SET blacklist:{access_token_jti} 1 EX {remaining_seconds}

  # For logout-all, store invalidation timestamp
  SET user_invalidated_at:{user_id} {timestamp}
```

**Complexity**: S
**Dependencies**: E1-US-004
**Status**: [ ] Not Started

---

## Phase E2: Storage Isolation & File Management

> **Priority**: 🔴 CRITICAL
> **Estimated Effort**: 2-3 days
> **Dependencies**: Phase E1

### E2-US-001: Per-User Storage Directory Creation

**Story**: As the system, when a user registers, I want to create their isolated storage directories, so that their files are separated from other users.

**Acceptance Criteria**:
- [ ] On user registration, create directory structure:
  ```
  /data/users/{user_id}/
      ├── documents/
      ├── emails/
      ├── chromadb/
      └── metadata.json
  ```
- [ ] Set filesystem permissions: owner=app, mode=700 (no other access)
- [ ] Initialize metadata.json with quota info
- [ ] Handle creation failures gracefully (rollback user creation)

**Technical Notes**:
```
Files to create/modify:
  backend/core/storage/__init__.py
  backend/core/storage/service.py     # Storage operations
  backend/core/storage/quota.py       # Quota management
  backend/core/auth/service.py        # Call storage setup on register

Directory Structure:
  DATA_ROOT=/data/users  # From environment variable

  def create_user_storage(user_id: str):
      user_dir = Path(DATA_ROOT) / user_id
      (user_dir / "documents").mkdir(parents=True, mode=0o700)
      (user_dir / "emails").mkdir(mode=0o700)
      (user_dir / "chromadb").mkdir(mode=0o700)

      metadata = {
          "user_id": user_id,
          "quota_bytes": 5 * 1024 * 1024 * 1024,  # 5GB
          "used_bytes": 0,
          "created_at": datetime.utcnow().isoformat()
      }
      (user_dir / "metadata.json").write_text(json.dumps(metadata))

Security:
  - Path traversal prevention: always resolve and validate paths
  - No symlinks allowed in user directories
```

**Complexity**: M
**Dependencies**: E1-US-001
**Status**: [ ] Not Started

---

### E2-US-002: 5GB Quota Enforcement

**Story**: As the system, I want to enforce a 5GB storage quota per user, so that resources are fairly distributed.

**Acceptance Criteria**:
- [ ] Track storage usage in metadata.json and database
- [ ] Before any file upload: check if new file would exceed quota
- [ ] If quota exceeded: reject upload with clear error message
- [ ] Recalculate usage when files are deleted
- [ ] Admin can adjust quota per user (future: E5)
- [ ] Show remaining space in upload UI

**Technical Notes**:
```
Files to modify:
  backend/core/storage/quota.py

Quota Check:
  def check_quota(user_id: str, file_size: int) -> bool:
      user = get_user(user_id)
      return (user.storage_used_bytes + file_size) <= user.storage_quota_bytes

Usage Calculation:
  def recalculate_usage(user_id: str) -> int:
      user_dir = Path(DATA_ROOT) / user_id
      total = sum(f.stat().st_size for f in user_dir.rglob("*") if f.is_file())
      # Exclude chromadb from quota (indexes don't count against user)
      return total - get_dir_size(user_dir / "chromadb")

API Response on Quota Exceeded:
  {
    "error": "Storage quota exceeded",
    "quota_bytes": 5368709120,
    "used_bytes": 5100000000,
    "file_size": 300000000,
    "available_bytes": 268709120
  }
```

**Complexity**: M
**Dependencies**: E2-US-001
**Status**: [ ] Not Started

---

### E2-US-003: Secure File Upload API

**Story**: As a user, I want to upload documents and email files, so that I can index and search them.

**Acceptance Criteria**:
- [ ] Upload endpoint accepts multipart/form-data
- [ ] Supported file types: .md, .txt, .docx, .eml, .pdf (future)
- [ ] File type validation (check magic bytes, not just extension)
- [ ] Maximum single file size: 100MB
- [ ] Quota check before accepting upload
- [ ] Store in appropriate subdirectory (documents/ or emails/)
- [ ] Return file metadata on success
- [ ] Handle duplicate filenames (add suffix: file(1).md)

**Technical Notes**:
```
Files to create/modify:
  backend/core/storage/upload.py      # Upload handling
  backend/core/storage/routes.py      # Upload endpoints
  agent_layer/src/routes/upload.js    # Proxy with multipart support

API Endpoint:
  POST /api/files/upload
  Content-Type: multipart/form-data
  Body: file (binary), type ("documents" | "emails")

  Response: {
    "success": true,
    "file": {
      "id": "uuid",
      "name": "document.md",
      "path": "documents/document.md",
      "size": 12345,
      "type": "documents",
      "uploaded_at": "2025-12-25T10:00:00Z"
    },
    "storage": {
      "used_bytes": 12345,
      "quota_bytes": 5368709120,
      "remaining_bytes": 5368696775
    }
  }

File Type Validation:
  import magic

  ALLOWED_TYPES = {
      '.md': 'text/plain',
      '.txt': 'text/plain',
      '.docx': 'application/vnd.openxmlformats-officedocument',
      '.eml': 'message/rfc822',
  }

  def validate_file_type(file_content: bytes, filename: str) -> bool:
      mime = magic.from_buffer(file_content, mime=True)
      ext = Path(filename).suffix.lower()
      return ext in ALLOWED_TYPES and ALLOWED_TYPES[ext] in mime
```

**Complexity**: L
**Dependencies**: E2-US-002
**Status**: [ ] Not Started

---

### E2-US-004: File Listing API

**Story**: As a user, I want to list all my uploaded files, so that I can manage my document library.

**Acceptance Criteria**:
- [ ] List all files in user's storage (documents + emails)
- [ ] Support filtering by type (documents, emails)
- [ ] Support sorting (name, size, date uploaded, date modified)
- [ ] Support pagination (default 50 per page)
- [ ] Return file metadata: name, size, type, uploaded_at, indexed (bool)
- [ ] Show indexing status for each file

**Technical Notes**:
```
API Endpoint:
  GET /api/files
  Query params:
    type: "documents" | "emails" | "all"
    sort: "name" | "size" | "uploaded_at" | "modified_at"
    order: "asc" | "desc"
    page: int
    limit: int (max 100)

  Response: {
    "files": [
      {
        "id": "uuid",
        "name": "doc.md",
        "path": "documents/doc.md",
        "size": 1234,
        "type": "documents",
        "uploaded_at": "...",
        "modified_at": "...",
        "indexed": true,
        "indexed_at": "..."
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 50,
      "total": 150,
      "pages": 3
    }
  }
```

**Complexity**: M
**Dependencies**: E2-US-003
**Status**: [ ] Not Started

---

### E2-US-005: File Deletion API

**Story**: As a user, I want to delete my uploaded files, so that I can manage storage and remove outdated documents.

**Acceptance Criteria**:
- [ ] Delete single file by ID or path
- [ ] Delete multiple files in batch
- [ ] Remove file from vector index when deleted
- [ ] Update storage usage after deletion
- [ ] Confirm deletion (UI shows confirmation dialog)
- [ ] Soft delete option with 30-day trash (future enhancement)

**Technical Notes**:
```
API Endpoints:
  DELETE /api/files/{file_id}
  Response: { success: true, storage: { used_bytes, remaining_bytes } }

  POST /api/files/delete-batch
  Body: { file_ids: ["uuid1", "uuid2"] }
  Response: {
    success: true,
    deleted: 2,
    failed: 0,
    storage: { used_bytes, remaining_bytes }
  }

Vector Index Cleanup:
  - When file deleted, queue job to remove its chunks from ChromaDB
  - Use document path as filter: chromadb.delete(where={"source": path})
```

**Complexity**: M
**Dependencies**: E2-US-004, E3-US-001 (for index cleanup job)
**Status**: [ ] Not Started

---

### E2-US-006: Storage Usage Dashboard

**Story**: As a user, I want to see my storage usage visually, so that I can understand how much space I have left.

**Acceptance Criteria**:
- [ ] Storage widget in header or sidebar (always visible)
- [ ] Progress bar showing used/total quota
- [ ] Breakdown by type (documents vs emails)
- [ ] List of largest files (top 5)
- [ ] Warning when approaching quota (>80%)
- [ ] Critical warning when nearly full (>95%)

**Technical Notes**:
```
UI Component:
  frontend/librarian-ui/components/storage-widget.js

  <div class="storage-widget">
    <div class="storage-bar">
      <div class="storage-used" style="width: 65%"></div>
    </div>
    <span class="storage-text">3.2 GB / 5 GB used</span>
    <div class="storage-breakdown">
      <span>Documents: 2.1 GB</span>
      <span>Emails: 1.1 GB</span>
    </div>
  </div>

API Endpoint:
  GET /api/storage/usage
  Response: {
    "total_bytes": 5368709120,
    "used_bytes": 3435973836,
    "available_bytes": 1932735284,
    "percentage": 64,
    "breakdown": {
      "documents": 2254857830,
      "emails": 1181116006
    },
    "largest_files": [
      { "name": "big-report.docx", "size": 52428800, "type": "documents" }
    ]
  }
```

**Complexity**: M
**Dependencies**: E2-US-004
**Status**: [ ] Not Started

---

## Phase E3: Background Indexing

> **Priority**: 🔴 CRITICAL - Solves 8-10 hour blocking issue
> **Estimated Effort**: 3-4 days
> **Dependencies**: Phase E1, E2

### E3-US-001: Job Queue Infrastructure

**Story**: As the system, I want a job queue for background tasks, so that indexing doesn't block the MCP server.

**Acceptance Criteria**:
- [ ] Redis-based job queue (or SQLite for simpler deployment)
- [ ] Jobs have: id, user_id, type, status, progress, created_at, started_at, completed_at
- [ ] Job statuses: pending, running, completed, failed, cancelled
- [ ] Jobs are processed in FIFO order per user
- [ ] Failed jobs can be retried (max 3 attempts)
- [ ] Job results stored for 24 hours

**Technical Notes**:
```
Files to create:
  backend/core/jobs/__init__.py
  backend/core/jobs/queue.py          # Job queue abstraction
  backend/core/jobs/models.py         # Job SQLAlchemy model
  backend/core/jobs/worker.py         # Worker process

Option A: Redis + RQ (Recommended for production)
  pip install rq redis

  # Start worker (separate process)
  rq worker --with-scheduler

Option B: SQLite Queue (Simpler, single-server)
  CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    job_type VARCHAR(50),  -- 'index_all', 'index_file', 'reindex'
    status VARCHAR(20) DEFAULT 'pending',
    progress FLOAT DEFAULT 0,  -- 0.0 to 1.0
    result JSONB,
    error TEXT,
    attempts INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
  );

Job Types:
  - index_all: Full index of user's documents
  - index_file: Index single new file
  - index_delta: Incremental index (changed files only)
  - reindex: Force reindex all
  - delete_from_index: Remove file from index
```

**Complexity**: L
**Dependencies**: E1-US-001
**Status**: [ ] Not Started

---

### E3-US-002: Indexing Worker Process

**Story**: As the system, I want separate worker processes for indexing, so that the MCP server remains responsive.

**Acceptance Criteria**:
- [ ] Worker process runs independently from MCP server
- [ ] Worker polls job queue for pending jobs
- [ ] Worker processes one job at a time (configurable concurrency)
- [ ] Worker updates job progress at regular intervals
- [ ] Worker handles crashes gracefully (job returns to queue)
- [ ] Multiple workers can run in parallel

**Technical Notes**:
```
Files to create:
  backend/worker.py                    # Worker entry point
  backend/core/jobs/handlers.py        # Job type handlers
  scripts/start_worker.sh              # Worker startup script

Worker Architecture:
  while True:
      job = queue.get_next_job()
      if job:
          try:
              job.status = 'running'
              job.started_at = datetime.utcnow()

              handler = get_handler(job.job_type)
              result = handler.execute(job, progress_callback)

              job.status = 'completed'
              job.result = result
          except Exception as e:
              job.attempts += 1
              if job.attempts >= 3:
                  job.status = 'failed'
                  job.error = str(e)
              else:
                  job.status = 'pending'  # Retry
      else:
          time.sleep(1)  # No jobs, wait

Indexing Handler:
  class IndexAllHandler:
      def execute(self, job, progress_callback):
          user_dir = get_user_storage(job.user_id)
          files = list(user_dir.rglob("*"))

          for i, file in enumerate(files):
              index_file(file, job.user_id)
              progress_callback(i / len(files))

          return {"files_indexed": len(files)}

Startup Script:
  #!/bin/bash
  cd /path/to/backend
  source venv/bin/activate
  python worker.py --concurrency 2
```

**Complexity**: L
**Dependencies**: E3-US-001
**Status**: [ ] Not Started

---

### E3-US-003: Index Job Creation API

**Story**: As a user, I want to trigger indexing of my documents, so that they become searchable.

**Acceptance Criteria**:
- [ ] "Index All" button triggers full index job
- [ ] Auto-trigger index job after file upload
- [ ] "Reindex" button forces full reindex
- [ ] Only one indexing job per user at a time
- [ ] Cancel existing job option
- [ ] Estimated time display based on file count

**Technical Notes**:
```
API Endpoints:
  POST /api/index/start
  Body: { type: "all" | "delta" | "reindex" }
  Response: {
    job_id: "uuid",
    status: "pending",
    estimated_minutes: 15,
    files_to_index: 150
  }

  POST /api/index/cancel
  Response: { success: true }

Auto-Trigger Logic (in upload handler):
  async def handle_upload(user_id, file):
      # Save file...

      # Queue single-file index job
      queue.enqueue({
          "user_id": user_id,
          "job_type": "index_file",
          "file_path": saved_path
      })

Prevent Duplicate Jobs:
  def start_index_job(user_id, job_type):
      existing = Job.query.filter_by(
          user_id=user_id,
          status__in=['pending', 'running']
      ).first()

      if existing:
          raise JobAlreadyRunning(existing.id)

      return create_job(user_id, job_type)
```

**Complexity**: M
**Dependencies**: E3-US-002
**Status**: [ ] Not Started

---

### E3-US-004: Job Status Tracking API

**Story**: As a user, I want to check the status of my indexing jobs, so that I know when my documents are searchable.

**Acceptance Criteria**:
- [ ] Get current job status and progress
- [ ] Get job history (last 10 jobs)
- [ ] Progress percentage (0-100)
- [ ] Estimated time remaining
- [ ] Files processed / total files count
- [ ] Error details if job failed

**Technical Notes**:
```
API Endpoints:
  GET /api/index/status
  Response: {
    "current_job": {
      "id": "uuid",
      "type": "index_all",
      "status": "running",
      "progress": 0.45,
      "files_processed": 67,
      "files_total": 150,
      "started_at": "...",
      "estimated_completion": "..."
    } | null,
    "last_indexed_at": "2025-12-25T10:00:00Z",
    "index_stats": {
      "documents": 120,
      "emails": 30,
      "chunks": 5400
    }
  }

  GET /api/index/history
  Response: {
    "jobs": [
      {
        "id": "uuid",
        "type": "index_all",
        "status": "completed",
        "progress": 1.0,
        "files_processed": 150,
        "duration_seconds": 847,
        "completed_at": "..."
      }
    ]
  }
```

**Complexity**: S
**Dependencies**: E3-US-003
**Status**: [ ] Not Started

---

### E3-US-005: Real-time Progress Updates (SSE)

**Story**: As a user watching the indexing progress, I want real-time updates, so that I don't have to refresh the page.

**Acceptance Criteria**:
- [ ] SSE endpoint streams progress updates
- [ ] Updates sent every 2-3 seconds during indexing
- [ ] Connection auto-reconnects if dropped
- [ ] UI updates progress bar and stats in real-time
- [ ] Notification when job completes or fails

**Technical Notes**:
```
Backend SSE Endpoint:
  GET /api/index/stream

  from fastapi.responses import StreamingResponse

  async def index_stream(user_id: str):
      async def event_generator():
          while True:
              job = get_current_job(user_id)
              if job:
                  yield f"data: {json.dumps(job.to_dict())}\n\n"
              await asyncio.sleep(2)

      return StreamingResponse(
          event_generator(),
          media_type="text/event-stream"
      )

Frontend:
  const eventSource = new EventSource('/api/index/stream');

  eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updateProgressBar(data.progress);
      updateStats(data.files_processed, data.files_total);

      if (data.status === 'completed') {
          showNotification('Indexing complete!');
          eventSource.close();
      }
  };

  eventSource.onerror = () => {
      // Auto-reconnect after 5 seconds
      setTimeout(() => connectSSE(), 5000);
  };
```

**Complexity**: M
**Dependencies**: E3-US-004
**Status**: [ ] Not Started

---

### E3-US-006: Indexing Status UI Component

**Story**: As a user, I want a visual indexing status widget, so that I can see progress at a glance.

**Acceptance Criteria**:
- [ ] Status widget shows: idle, indexing, completed, failed
- [ ] Progress bar during indexing
- [ ] Files processed counter
- [ ] Estimated time remaining
- [ ] Last indexed timestamp when idle
- [ ] "Start Indexing" button when no job running
- [ ] "Cancel" button during indexing

**Technical Notes**:
```
UI Component:
  frontend/librarian-ui/components/indexing-status.js

  States:
    1. Idle:
       "Last indexed: 2 hours ago"
       [Start Indexing] [Reindex]

    2. Pending:
       "Indexing queued... Position: 1"
       [Cancel]

    3. Running:
       ████████░░░░░░░░ 45%
       "Processing: 67 / 150 files"
       "ETA: ~5 minutes"
       [Cancel]

    4. Completed:
       ✓ "Indexing complete"
       "150 files indexed in 14 minutes"
       [Dismiss]

    5. Failed:
       ✗ "Indexing failed"
       "Error: Out of memory"
       [Retry] [View Details]
```

**Complexity**: M
**Dependencies**: E3-US-005
**Status**: [ ] Not Started

---

### E3-US-007: Incremental Indexing Support

**Story**: As a user who uploads new files regularly, I want only new/changed files to be indexed, so that indexing is fast.

**Acceptance Criteria**:
- [ ] Track file hashes to detect changes
- [ ] Delta index only processes new/modified files
- [ ] Detect deleted files and remove from index
- [ ] Option to force full reindex if needed
- [ ] Show "X new files to index" when changes detected

**Technical Notes**:
```
File Hash Tracking:
  CREATE TABLE file_hashes (
    user_id UUID REFERENCES users(id),
    file_path VARCHAR(500),
    content_hash VARCHAR(64),  -- SHA-256
    indexed_at TIMESTAMP,
    PRIMARY KEY (user_id, file_path)
  );

Delta Detection:
  def get_files_to_index(user_id: str) -> List[Path]:
      user_dir = get_user_storage(user_id)
      current_files = {f: hash_file(f) for f in user_dir.rglob("*") if f.is_file()}
      stored_hashes = get_stored_hashes(user_id)

      to_index = []
      for path, hash in current_files.items():
          if path not in stored_hashes or stored_hashes[path] != hash:
              to_index.append(path)

      # Also handle deletions
      deleted = set(stored_hashes.keys()) - set(current_files.keys())
      remove_from_index(user_id, deleted)

      return to_index
```

**Complexity**: M
**Dependencies**: E3-US-002
**Status**: [ ] Not Started

---

### E3-US-008: Index Rebuild on Demand

**Story**: As a user, I want to force a complete index rebuild, so that I can fix any index corruption issues.

**Acceptance Criteria**:
- [ ] "Rebuild Index" button in settings or admin area
- [ ] Confirmation dialog (destructive operation)
- [ ] Deletes existing vector index completely
- [ ] Re-indexes all files from scratch
- [ ] Preserves file hashes for future delta indexing
- [ ] Shows warning about time required

**Technical Notes**:
```
Rebuild Process:
  1. User confirms rebuild
  2. Delete user's ChromaDB collection
  3. Clear file_hashes table for user
  4. Queue 'index_all' job
  5. Track progress as normal

API Endpoint:
  POST /api/index/rebuild
  Body: { confirm: true }
  Response: { job_id: "uuid", warning: "This may take 15-30 minutes" }

ChromaDB Collection Deletion:
  def delete_user_index(user_id: str):
      collection_name = f"user_{user_id}"
      chroma_client.delete_collection(collection_name)
```

**Complexity**: S
**Dependencies**: E3-US-003
**Status**: [ ] Not Started

---

## Phase E4: Multi-Tenant Search

> **Priority**: 🟡 HIGH
> **Estimated Effort**: 2 days
> **Dependencies**: Phase E1, E2, E3

### E4-US-001: Per-User Vector Database Collections

**Story**: As the system, I want separate ChromaDB collections per user, so that search data is completely isolated.

**Acceptance Criteria**:
- [ ] Each user gets their own ChromaDB collection
- [ ] Collection name format: `user_{user_id}`
- [ ] Collections stored in user's storage directory
- [ ] Search queries only access user's own collection
- [ ] Admin can optionally access any collection

**Technical Notes**:
```
Files to modify:
  backend/core/vector_db.py           # Multi-tenant support
  backend/core/hybrid_search.py       # User context in search

Collection Management:
  class MultiTenantVectorDB:
      def __init__(self, data_root: str):
          self.data_root = Path(data_root)
          self.clients = {}  # user_id -> chromadb.Client

      def get_collection(self, user_id: str):
          if user_id not in self.clients:
              persist_dir = self.data_root / user_id / "chromadb"
              self.clients[user_id] = chromadb.PersistentClient(
                  path=str(persist_dir)
              )

          return self.clients[user_id].get_or_create_collection(
              name=f"documents",
              embedding_function=self.embedding_fn
          )

Indexing with User Context:
  def index_document(user_id: str, doc_path: Path, chunks: List[str]):
      collection = vector_db.get_collection(user_id)
      collection.add(
          documents=chunks,
          metadatas=[{"source": str(doc_path)} for _ in chunks],
          ids=[f"{doc_path}_{i}" for i in range(len(chunks))]
      )
```

**Complexity**: M
**Dependencies**: E3-US-002
**Status**: [ ] Not Started

---

### E4-US-002: Tenant Context in Search API

**Story**: As a logged-in user, I want my searches to only return my documents, so that I don't see other users' data.

**Acceptance Criteria**:
- [ ] All search endpoints require authentication
- [ ] User ID extracted from JWT token
- [ ] Search only queries user's collection
- [ ] MCP tools receive user context
- [ ] Unauthorized access attempts logged

**Technical Notes**:
```
Files to modify:
  backend/mcp_server/tools.py         # Add user context
  backend/core/hybrid_search.py       # Tenant-aware search
  agent_layer/src/routes/search.js    # Pass user context

Search with User Context:
  @mcp.tool()
  def search_documentation(
      query: str,
      user_id: str,  # Injected from auth middleware
      ...
  ) -> dict:
      # Search only user's collection
      results = search_engine.search(
          query=query,
          collection=f"user_{user_id}",
          ...
      )
      return results

Agent Layer Middleware:
  // auth.js middleware
  req.user = verifyJWT(req.cookies.access_token);

  // search.js route
  app.post('/api/search', auth, async (req, res) => {
      const results = await mcpClient.search({
          ...req.body,
          user_id: req.user.id  // Inject user context
      });
      res.json(results);
  });
```

**Complexity**: M
**Dependencies**: E4-US-001, E1-US-004
**Status**: [ ] Not Started

---

### E4-US-003: Search Result Isolation Validation

**Story**: As the system, I want to validate that search results belong to the requesting user, so that data leakage is prevented.

**Acceptance Criteria**:
- [ ] All returned document paths validated against user's storage
- [ ] Cross-tenant access attempts blocked and logged
- [ ] Security audit log for any anomalies
- [ ] Automated tests for isolation

**Technical Notes**:
```
Validation Layer:
  def validate_results(user_id: str, results: List[dict]) -> List[dict]:
      user_prefix = f"/data/users/{user_id}/"
      validated = []

      for result in results:
          if result['file_path'].startswith(user_prefix):
              validated.append(result)
          else:
              # SECURITY ALERT - log this
              logger.critical(f"Cross-tenant access attempt: {user_id} -> {result['file_path']}")
              audit_log.record(
                  event="cross_tenant_access_blocked",
                  user_id=user_id,
                  attempted_path=result['file_path']
              )

      return validated

Automated Security Tests:
  def test_search_isolation():
      # Create two users
      user_a = create_test_user("a@test.com")
      user_b = create_test_user("b@test.com")

      # User A uploads a document
      upload_document(user_a, "secret.md", "User A's secret data")
      index_documents(user_a)

      # User B searches - should NOT find User A's document
      results = search_as_user(user_b, "secret data")
      assert len(results) == 0

      # User A searches - SHOULD find their document
      results = search_as_user(user_a, "secret data")
      assert len(results) == 1
```

**Complexity**: M
**Dependencies**: E4-US-002
**Status**: [ ] Not Started

---

### E4-US-004: Shared Documentation Space (Optional)

**Story**: As an organization, I want to provide shared documentation that all users can search, so that company-wide knowledge is accessible.

**Acceptance Criteria**:
- [ ] Shared docs stored in `/data/shared/`
- [ ] Indexed in a `shared` collection
- [ ] All authenticated users can search shared docs
- [ ] Search UI toggle: "Include shared docs"
- [ ] Results clearly marked as "Shared" vs "Personal"
- [ ] Admin-only upload to shared space

**Technical Notes**:
```
Shared Collection:
  SHARED_COLLECTION = "shared_documentation"

Search with Shared:
  def search(user_id: str, query: str, include_shared: bool = True):
      collections_to_search = [f"user_{user_id}"]

      if include_shared:
          collections_to_search.append(SHARED_COLLECTION)

      all_results = []
      for collection in collections_to_search:
          results = search_collection(collection, query)
          for r in results:
              r['source'] = 'shared' if collection == SHARED_COLLECTION else 'personal'
          all_results.extend(results)

      return sorted(all_results, key=lambda x: x['score'], reverse=True)

UI Toggle:
  <label>
    <input type="checkbox" id="includeShared" checked>
    Include shared documentation
  </label>
```

**Complexity**: M
**Dependencies**: E4-US-002
**Status**: [ ] Not Started

---

## Phase E5: Admin Dashboard

> **Priority**: 🟢 MEDIUM
> **Estimated Effort**: 2-3 days
> **Dependencies**: Phase E1

### E5-US-001: Admin Role & Permissions

**Story**: As the system owner, I want admin users with elevated permissions, so that I can manage the platform.

**Acceptance Criteria**:
- [ ] Admin role in user table (role='admin')
- [ ] First registered user automatically becomes admin
- [ ] Admins can access all admin endpoints
- [ ] Admin UI section with access control
- [ ] Regular users denied access to admin endpoints

**Technical Notes**:
```
Role Check Decorator:
  def require_admin(func):
      @wraps(func)
      async def wrapper(request: Request, *args, **kwargs):
          user = request.state.user
          if user.role != 'admin':
              raise HTTPException(403, "Admin access required")
          return await func(request, *args, **kwargs)
      return wrapper

Admin Routes:
  @router.get("/admin/users")
  @require_admin
  async def list_users():
      return User.query.all()
```

**Complexity**: S
**Dependencies**: E1-US-002
**Status**: [ ] Not Started

---

### E5-US-002: User Management CRUD

**Story**: As an admin, I want to manage user accounts, so that I can help users and maintain the platform.

**Acceptance Criteria**:
- [ ] List all users with pagination and search
- [ ] View user details (profile, storage usage, last login)
- [ ] Edit user (display name, role, quota)
- [ ] Disable/enable user accounts
- [ ] Delete user (with data cleanup option)
- [ ] Reset user password (generates reset link)

**Technical Notes**:
```
API Endpoints:
  GET    /api/admin/users                  # List users
  GET    /api/admin/users/{id}             # Get user details
  PATCH  /api/admin/users/{id}             # Update user
  DELETE /api/admin/users/{id}             # Delete user
  POST   /api/admin/users/{id}/reset-password
  POST   /api/admin/users/{id}/disable
  POST   /api/admin/users/{id}/enable

User List Response:
  {
    "users": [
      {
        "id": "uuid",
        "email": "user@example.com",
        "display_name": "John Doe",
        "role": "user",
        "storage_used_bytes": 1234567,
        "storage_quota_bytes": 5368709120,
        "created_at": "...",
        "last_login_at": "...",
        "is_active": true,
        "indexing_status": "idle"
      }
    ],
    "pagination": { ... }
  }
```

**Complexity**: L
**Dependencies**: E5-US-001
**Status**: [ ] Not Started

---

### E5-US-003: System Health Dashboard

**Story**: As an admin, I want to see system health metrics, so that I can monitor platform performance.

**Acceptance Criteria**:
- [ ] Total users count and active users (last 7 days)
- [ ] Total storage used across all users
- [ ] Active indexing jobs count
- [ ] Job queue depth
- [ ] Worker status (running, idle, failed)
- [ ] Database connection status
- [ ] Redis connection status
- [ ] MCP server status

**Technical Notes**:
```
API Endpoint:
  GET /api/admin/health

Response:
  {
    "status": "healthy",
    "users": {
      "total": 150,
      "active_7d": 89
    },
    "storage": {
      "total_bytes": 107374182400,  # 100 GB
      "used_bytes": 53687091200,    # 50 GB
      "percentage": 50
    },
    "jobs": {
      "pending": 3,
      "running": 2,
      "failed_24h": 1
    },
    "workers": {
      "active": 2,
      "idle": 0
    },
    "services": {
      "database": "connected",
      "redis": "connected",
      "mcp_backend": "healthy"
    },
    "uptime_seconds": 86400
  }

UI Dashboard:
  frontend/librarian-ui/admin/health.html
  - Metric cards with values and trends
  - Service status indicators (green/red)
  - Auto-refresh every 30 seconds
```

**Complexity**: M
**Dependencies**: E5-US-001
**Status**: [ ] Not Started

---

### E5-US-004: Per-User Usage Analytics

**Story**: As an admin, I want to see per-user analytics, so that I can understand platform usage patterns.

**Acceptance Criteria**:
- [ ] Per-user metrics: searches, documents, storage, last activity
- [ ] Search volume over time (daily/weekly/monthly)
- [ ] Most active users ranking
- [ ] Users approaching quota alerts
- [ ] Export usage report (CSV)

**Technical Notes**:
```
Analytics Tables:
  CREATE TABLE search_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    query TEXT,
    result_count INT,
    search_mode VARCHAR(20),
    duration_ms INT,
    created_at TIMESTAMP DEFAULT NOW()
  );

API Endpoints:
  GET /api/admin/analytics/overview
  GET /api/admin/analytics/users/{id}
  GET /api/admin/analytics/export?format=csv

Overview Response:
  {
    "period": "last_30_days",
    "searches": {
      "total": 15000,
      "daily_avg": 500,
      "trend": "+12%"
    },
    "top_users": [
      { "email": "...", "searches": 1500 }
    ],
    "quota_warnings": [
      { "email": "...", "usage_percentage": 92 }
    ]
  }
```

**Complexity**: M
**Dependencies**: E5-US-002
**Status**: [ ] Not Started

---

### E5-US-005: Indexing Job Management

**Story**: As an admin, I want to manage all indexing jobs, so that I can troubleshoot and optimize.

**Acceptance Criteria**:
- [ ] View all jobs across all users
- [ ] Filter by status (pending, running, failed)
- [ ] Cancel any running job
- [ ] Retry failed jobs
- [ ] View job logs/errors
- [ ] Prioritize specific user's job

**Technical Notes**:
```
API Endpoints:
  GET    /api/admin/jobs                    # List all jobs
  GET    /api/admin/jobs/{id}               # Job details
  POST   /api/admin/jobs/{id}/cancel
  POST   /api/admin/jobs/{id}/retry
  POST   /api/admin/jobs/{id}/prioritize

Job Details Response:
  {
    "id": "uuid",
    "user": { "id": "...", "email": "..." },
    "type": "index_all",
    "status": "failed",
    "progress": 0.67,
    "files_processed": 100,
    "files_total": 150,
    "error": "Out of memory while processing large.docx",
    "logs": [
      { "time": "...", "message": "Started indexing" },
      { "time": "...", "message": "Error: ..." }
    ],
    "attempts": 2,
    "created_at": "...",
    "started_at": "...",
    "failed_at": "..."
  }
```

**Complexity**: M
**Dependencies**: E5-US-001, E3-US-004
**Status**: [ ] Not Started

---

## Phase E6: Advanced Features

> **Priority**: 🟢 NICE-TO-HAVE
> **Estimated Effort**: 3-4 days
> **Dependencies**: All previous phases

### E6-US-001: Saved Searches

**Story**: As a user, I want to save frequently used searches, so that I can quickly run them again.

**Acceptance Criteria**:
- [ ] "Save Search" button on results page
- [ ] Name the saved search
- [ ] Saved searches list in sidebar or dropdown
- [ ] Click to execute saved search
- [ ] Edit/delete saved searches
- [ ] Maximum 50 saved searches per user

**Technical Notes**:
```
Database Schema:
  CREATE TABLE saved_searches (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(100),
    query TEXT,
    search_type VARCHAR(20),  -- emails, documentation
    search_mode VARCHAR(20),  -- auto, keyword, semantic, etc.
    filters JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP
  );

API Endpoints:
  GET    /api/searches/saved
  POST   /api/searches/saved
  DELETE /api/searches/saved/{id}
  POST   /api/searches/saved/{id}/execute
```

**Complexity**: M
**Dependencies**: E4-US-002
**Status**: [ ] Not Started

---

### E6-US-002: Search History

**Story**: As a user, I want to see my recent searches, so that I can revisit previous queries.

**Acceptance Criteria**:
- [ ] Last 100 searches stored per user
- [ ] Recent searches dropdown on search input focus
- [ ] Click to re-run search
- [ ] Clear history option
- [ ] Privacy: history only visible to user

**Technical Notes**:
```
Database Schema:
  CREATE TABLE search_history (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    query TEXT,
    search_type VARCHAR(20),
    result_count INT,
    created_at TIMESTAMP DEFAULT NOW()
  );

  -- Index for fast retrieval
  CREATE INDEX idx_search_history_user_time
    ON search_history(user_id, created_at DESC);

API Endpoints:
  GET    /api/searches/history?limit=10
  DELETE /api/searches/history  # Clear all
```

**Complexity**: S
**Dependencies**: E4-US-002
**Status**: [ ] Not Started

---

### E6-US-003: Document Bookmarks

**Story**: As a user, I want to bookmark important documents, so that I can quickly access them.

**Acceptance Criteria**:
- [ ] Bookmark icon on search results and document view
- [ ] Bookmarks list accessible from sidebar
- [ ] Organize bookmarks in folders (optional)
- [ ] Maximum 500 bookmarks per user
- [ ] Bookmark includes note field

**Technical Notes**:
```
Database Schema:
  CREATE TABLE bookmarks (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    file_path VARCHAR(500),
    title VARCHAR(200),
    note TEXT,
    folder VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
  );

API Endpoints:
  GET    /api/bookmarks
  POST   /api/bookmarks
  PATCH  /api/bookmarks/{id}
  DELETE /api/bookmarks/{id}
```

**Complexity**: M
**Dependencies**: E4-US-002
**Status**: [ ] Not Started

---

### E6-US-004: Audit Logging

**Story**: As an admin, I want comprehensive audit logs, so that I can track all important actions.

**Acceptance Criteria**:
- [ ] Log: login, logout, failed login, password change
- [ ] Log: file upload, delete
- [ ] Log: index start, complete, fail
- [ ] Log: admin actions (user management)
- [ ] Retention: 90 days
- [ ] Export: CSV download
- [ ] Search/filter logs

**Technical Notes**:
```
Database Schema:
  CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID,
    action VARCHAR(50),
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
  );

  -- Partition by month for performance
  -- Automatic cleanup of logs > 90 days

Action Types:
  - auth.login, auth.logout, auth.login_failed
  - auth.password_change, auth.password_reset
  - file.upload, file.delete
  - index.start, index.complete, index.fail
  - admin.user_create, admin.user_update, admin.user_delete
  - admin.quota_change

API Endpoint (Admin only):
  GET /api/admin/audit-logs?action=auth.login&from=2025-12-01
```

**Complexity**: M
**Dependencies**: E5-US-001
**Status**: [ ] Not Started

---

### E6-US-005: API Rate Limiting

**Story**: As the system, I want to rate limit API requests, so that no single user can overwhelm the system.

**Acceptance Criteria**:
- [ ] Rate limits per user per endpoint category
- [ ] Default: 100 requests/minute for search
- [ ] Default: 20 requests/minute for uploads
- [ ] 429 Too Many Requests response when exceeded
- [ ] Rate limit headers in response (X-RateLimit-*)
- [ ] Admin can adjust per-user limits

**Technical Notes**:
```
Rate Limit Categories:
  - search: 100/minute
  - upload: 20/minute
  - auth: 10/minute
  - admin: 60/minute

Redis-based Limiting:
  from fastapi_limiter import FastAPILimiter
  from fastapi_limiter.depends import RateLimiter

  @router.post("/search")
  @limiter.limit("100/minute")
  async def search(request: Request):
      ...

Response Headers:
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1703500060
```

**Complexity**: M
**Dependencies**: E1-US-004
**Status**: [ ] Not Started

---

### E6-US-006: Webhook Notifications

**Story**: As a power user, I want webhook notifications for indexing events, so that I can integrate with other systems.

**Acceptance Criteria**:
- [ ] Configure webhook URL in user settings
- [ ] Events: index.started, index.completed, index.failed
- [ ] Webhook includes: event type, timestamp, job details
- [ ] Retry failed webhooks (3 attempts)
- [ ] Webhook secret for signature verification
- [ ] Test webhook button

**Technical Notes**:
```
Database Schema:
  CREATE TABLE webhooks (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    url VARCHAR(500),
    secret VARCHAR(100),
    events TEXT[],  -- ['index.completed', 'index.failed']
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
  );

Webhook Payload:
  {
    "event": "index.completed",
    "timestamp": "2025-12-25T12:00:00Z",
    "data": {
      "job_id": "uuid",
      "files_indexed": 150,
      "duration_seconds": 847
    }
  }

Signature Header:
  X-Webhook-Signature: sha256=<HMAC-SHA256(secret, payload)>

API Endpoints:
  GET    /api/webhooks
  POST   /api/webhooks
  PATCH  /api/webhooks/{id}
  DELETE /api/webhooks/{id}
  POST   /api/webhooks/{id}/test
```

**Complexity**: L
**Dependencies**: E3-US-003
**Status**: [ ] Not Started

---

## Technical Appendix

### A. Database Schema (Complete)

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    storage_quota_bytes BIGINT DEFAULT 5368709120,
    storage_used_bytes BIGINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Password Reset Tokens
CREATE TABLE password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexing Jobs
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    progress FLOAT DEFAULT 0,
    files_total INT DEFAULT 0,
    files_processed INT DEFAULT 0,
    result JSONB,
    error TEXT,
    attempts INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- File Hashes (for incremental indexing)
CREATE TABLE file_hashes (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    file_size BIGINT,
    indexed_at TIMESTAMP,
    PRIMARY KEY (user_id, file_path)
);

-- Saved Searches
CREATE TABLE saved_searches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    query TEXT NOT NULL,
    search_type VARCHAR(20),
    search_mode VARCHAR(20),
    filters JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used_at TIMESTAMP
);

-- Search History
CREATE TABLE search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    search_type VARCHAR(20),
    result_count INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Bookmarks
CREATE TABLE bookmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    title VARCHAR(200),
    note TEXT,
    folder VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit Logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Webhooks
CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    url VARCHAR(500) NOT NULL,
    secret VARCHAR(100),
    events TEXT[] NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_jobs_user_status ON jobs(user_id, status);
CREATE INDEX idx_search_history_user_time ON search_history(user_id, created_at DESC);
CREATE INDEX idx_audit_logs_user_action ON audit_logs(user_id, action, created_at DESC);
```

### B. API Endpoint Summary

| Category | Method | Endpoint | Auth | Admin |
|----------|--------|----------|------|-------|
| Auth | POST | /api/auth/register | No | No |
| Auth | POST | /api/auth/login | No | No |
| Auth | POST | /api/auth/logout | Yes | No |
| Auth | POST | /api/auth/refresh | Cookie | No |
| Auth | POST | /api/auth/forgot-password | No | No |
| Auth | POST | /api/auth/reset-password | No | No |
| User | GET | /api/user/profile | Yes | No |
| User | PATCH | /api/user/profile | Yes | No |
| User | POST | /api/user/change-password | Yes | No |
| Files | GET | /api/files | Yes | No |
| Files | POST | /api/files/upload | Yes | No |
| Files | DELETE | /api/files/{id} | Yes | No |
| Files | POST | /api/files/delete-batch | Yes | No |
| Storage | GET | /api/storage/usage | Yes | No |
| Index | POST | /api/index/start | Yes | No |
| Index | POST | /api/index/cancel | Yes | No |
| Index | GET | /api/index/status | Yes | No |
| Index | GET | /api/index/stream | Yes | No |
| Index | GET | /api/index/history | Yes | No |
| Index | POST | /api/index/rebuild | Yes | No |
| Search | POST | /api/search | Yes | No |
| Search | POST | /api/document | Yes | No |
| Searches | GET | /api/searches/saved | Yes | No |
| Searches | POST | /api/searches/saved | Yes | No |
| Searches | DELETE | /api/searches/saved/{id} | Yes | No |
| Searches | GET | /api/searches/history | Yes | No |
| Bookmarks | GET | /api/bookmarks | Yes | No |
| Bookmarks | POST | /api/bookmarks | Yes | No |
| Bookmarks | DELETE | /api/bookmarks/{id} | Yes | No |
| Webhooks | GET | /api/webhooks | Yes | No |
| Webhooks | POST | /api/webhooks | Yes | No |
| Webhooks | POST | /api/webhooks/{id}/test | Yes | No |
| Admin | GET | /api/admin/users | Yes | Yes |
| Admin | GET | /api/admin/users/{id} | Yes | Yes |
| Admin | PATCH | /api/admin/users/{id} | Yes | Yes |
| Admin | DELETE | /api/admin/users/{id} | Yes | Yes |
| Admin | GET | /api/admin/health | Yes | Yes |
| Admin | GET | /api/admin/analytics/overview | Yes | Yes |
| Admin | GET | /api/admin/jobs | Yes | Yes |
| Admin | POST | /api/admin/jobs/{id}/cancel | Yes | Yes |
| Admin | GET | /api/admin/audit-logs | Yes | Yes |

### C. Security Checklist

- [ ] All passwords hashed with bcrypt (work factor 12)
- [ ] JWT tokens with short expiry + refresh tokens
- [ ] httpOnly, Secure, SameSite cookies
- [ ] HTTPS only in production
- [ ] Rate limiting on all endpoints
- [ ] CORS configured for specific origins
- [ ] Path traversal prevention in file operations
- [ ] SQL injection prevention (parameterized queries/ORM)
- [ ] XSS prevention (escape all output)
- [ ] File type validation (magic bytes, not extension)
- [ ] Maximum file size limits
- [ ] Storage quota enforcement
- [ ] Cross-tenant isolation in search
- [ ] Audit logging for sensitive operations
- [ ] Account lockout after failed logins
- [ ] Secure password reset flow

### D. Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/librarian
DATABASE_POOL_SIZE=10

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET_KEY=<random-256-bit-key>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Storage
DATA_ROOT=/data/users
SHARED_DOCS_ROOT=/data/shared
MAX_UPLOAD_SIZE_MB=100
DEFAULT_QUOTA_GB=5

# Email (for password reset)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=noreply@example.com
SMTP_PASSWORD=<password>
FROM_EMAIL=Librarian <noreply@example.com>

# Workers
WORKER_CONCURRENCY=2
JOB_TIMEOUT_SECONDS=3600

# Security
CORS_ORIGINS=http://localhost:8080,https://librarian.example.com
RATE_LIMIT_ENABLED=true
```

---

## Development Tracking

### Status Summary

| Phase | Stories | Not Started | In Progress | Complete |
|-------|---------|-------------|-------------|----------|
| E1: Auth & Users | 6 | 6 | 0 | 0 |
| E2: Storage | 6 | 6 | 0 | 0 |
| E3: Background Indexing | 8 | 8 | 0 | 0 |
| E4: Multi-Tenant Search | 4 | 4 | 0 | 0 |
| E5: Admin Dashboard | 5 | 5 | 0 | 0 |
| E6: Advanced Features | 6 | 6 | 0 | 0 |
| **Total** | **35** | **35** | **0** | **0** |

### Priority Order

1. **E1**: Foundation - all other features depend on this
2. **E3**: Critical for usability (8-10 hour blocking)
3. **E2**: Required for multi-user data management
4. **E4**: Core multi-tenant functionality
5. **E5**: Operational necessities
6. **E6**: Nice-to-have enhancements

### Changelog

| Date | Change | Stories Affected |
|------|--------|------------------|
| 2025-12-25 | Initial specification created | All |

---

## AI Development Notes

### For Claude Code / Codex / Gemini CLI

**Project Context**:
- Existing codebase: Librarian MCP v2.2.0 (single-user)
- Backend: Python 3.11+, FastAPI, ChromaDB, sentence-transformers
- Agent Layer: Node.js 18+, Express
- Frontend: Vanilla JavaScript (no framework)
- See CLAUDE.md for complete technical documentation

**Development Order**:
1. Start with E1-US-001 (User Registration)
2. Follow dependency chain in each phase
3. Each story is self-contained and testable
4. Run existing tests after each change: `pytest backend/tests/`

**File Organization**:
```
backend/
  core/
    auth/           # New - E1
    storage/        # New - E2
    jobs/           # New - E3
  models/           # New - SQLAlchemy models
  worker.py         # New - E3

agent_layer/
  src/
    middleware/
      auth.js       # New - E1
    routes/
      auth.js       # New - E1
      upload.js     # New - E2

frontend/
  librarian-ui/
    login.html      # New - E1
    register.html   # New - E1
    profile.html    # New - E1
    admin/          # New - E5
    components/
      storage-widget.js    # New - E2
      indexing-status.js   # New - E3
```

**Testing Strategy**:
- Unit tests for each new module
- Integration tests for API endpoints
- Security tests for isolation (E4-US-003)
- Load tests for rate limiting

---

*Document generated by Claude Code with ultrathink analysis*
*Last updated: 2025-12-25*
