import os
import re
import sqlite3
import datetime
from contextlib import contextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from passlib.context import CryptContext

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "colmado.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

# ---------- App ----------
app = FastAPI(title="Colmado Inventory System")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Security ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------- DB Helpers ----------
@contextmanager
def db_conn():
    """
    Context-managed SQLite connection that:
    - enables foreign keys
    - always closes
    - commits on success, rolls back on error
    """
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """
    SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we check schema first.
    Adds a small identifier validation so ALTER TABLE isn't injectable.
    """
    if not _identifier_re.match(table) or not _identifier_re.match(column):
        raise ValueError("Invalid table/column identifier")

    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


def _utc_iso_now() -> str:
    # Consistent UTC timestamps stored as ISO strings without tzinfo
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()


def migrate_purchases_table_if_needed(conn: sqlite3.Connection) -> None:
    """
    Fix older schemas where purchases.employee_id was NOT NULL DEFAULT 0 with FK to users(id).
    Under PRAGMA foreign_keys=ON, employee_id=0 breaks FK constraints.

    Migration strategy:
    - rebuild purchases table so employee_id is NULLable
    - map employee_id=0 -> NULL
    """
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='purchases'")
    if not cur.fetchone():
        return

    cur.execute("PRAGMA table_info(purchases)")
    cols = cur.fetchall()
    info = {c["name"]: dict(c) for c in cols}

    if "employee_id" not in info:
        return

    emp = info["employee_id"]
    notnull = int(emp.get("notnull", 0)) == 1

    # If it's already nullable, no rebuild needed.
    if not notnull:
        return

    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS purchases_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL,
                employee_id INTEGER,                 -- NULLable
                qty INTEGER NOT NULL,
                cost REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(product_id) REFERENCES products(id),
                FOREIGN KEY(employee_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)

        cur.execute("""
            INSERT INTO purchases_new (id, product_id, employee_id, qty, cost, created_at)
            SELECT
                id,
                product_id,
                CASE WHEN employee_id = 0 THEN NULL ELSE employee_id END AS employee_id,
                qty,
                COALESCE(cost, 0),
                created_at
            FROM purchases
        """)

        cur.execute("DROP TABLE purchases")
        cur.execute("ALTER TABLE purchases_new RENAME TO purchases")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_purchases_created_at ON purchases(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_purchases_employee_id ON purchases(employee_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_purchases_product_id ON purchases(product_id)")
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def init_db():
    with db_conn() as conn:
        cur = conn.cursor()

        # USERS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL,                 -- 'manager' or 'employee'
            unit_id TEXT NOT NULL DEFAULT '',
            pin_hash TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """)
        ensure_column(conn, "users", "can_purchase", "INTEGER NOT NULL DEFAULT 0")

        # PRODUCTS
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL DEFAULT '',
            name_en TEXT NOT NULL DEFAULT '',
            name_es TEXT NOT NULL DEFAULT '',
            sku TEXT NOT NULL DEFAULT '',
            qty INTEGER NOT NULL DEFAULT 0,
            reorder_level INTEGER NOT NULL DEFAULT 0,
            price REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """)

        # SALES
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            employee_id INTEGER NOT NULL,
            qty INTEGER NOT NULL,
            price REAL NOT NULL DEFAULT 0,      -- snapshot at time of sale
            created_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(id),
            FOREIGN KEY(employee_id) REFERENCES users(id)
        )
        """)

        # PURCHASES (employee_id NULLable)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS purchases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            employee_id INTEGER,                -- NULLable
            qty INTEGER NOT NULL,
            cost REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(id),
            FOREIGN KEY(employee_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """)

        # Migrate older schema if needed + ensure column exists for very old schemas
        migrate_purchases_table_if_needed(conn)
        ensure_column(conn, "purchases", "employee_id", "INTEGER")

        # CASH
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cash (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
        """)

        # Seed manager if missing
        cur.execute("SELECT id FROM users WHERE username = ?", ("manager",))
        row = cur.fetchone()
        if not row:
            now = _utc_iso_now()
            cur.execute("""
                INSERT INTO users (username, role, unit_id, pin_hash, active, can_purchase, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("manager", "manager", "M-001", pwd_context.hash("123456"), 1, 1, now))

        # Helpful indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_created_at ON sales(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_employee_id ON sales(employee_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_product_id ON sales(product_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_products_category_name ON products(category, name_es, name_en)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku)")


init_db()

# ---------- Auth (simple header token) ----------
# Token == username for simplicity.
# Client sends: X-User: <username>
def require_user(request: Request):
    token = request.headers.get("X-User", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND active=1", (token,))
        u = cur.fetchone()

    if not u:
        raise HTTPException(status_code=401, detail="Invalid user")

    return dict(u)


def require_manager(user=Depends(require_user)):
    if user["role"] != "manager":
        raise HTTPException(status_code=403, detail="Manager only")
    return user


def require_purchase_permission(user=Depends(require_user)):
    if user["role"] == "manager":
        return user
    if int(user.get("can_purchase", 0)) == 1:
        return user
    raise HTTPException(status_code=403, detail="Not allowed to add inventory")


# ---------- Models ----------
class LoginIn(BaseModel):
    username: str
    pin: str


class SaleIn(BaseModel):
    product_id: int
    qty: int


class CashIn(BaseModel):
    amount: float
    note: Optional[str] = ""


class UserCreateIn(BaseModel):
    username: str
    pin: str
    unit_id: str
    can_purchase: Optional[int] = 0


class SetPurchasePermIn(BaseModel):
    user_id: int
    can_purchase: int  # 0 or 1


class ChangePinIn(BaseModel):
    old_pin: str
    new_pin: str


class PurchaseLineIn(BaseModel):
    product_id: int
    qty: int
    cost: Optional[float] = 0


class PurchaseBulkIn(BaseModel):
    items: List[PurchaseLineIn]


# Product management (recommended)
class ProductCreateIn(BaseModel):
    category: Optional[str] = ""
    name_en: str
    name_es: str
    sku: str
    qty: Optional[int] = 0
    reorder_level: Optional[int] = 0
    price: Optional[float] = 0


class ProductUpdateIn(BaseModel):
    category: Optional[str] = None
    name_en: Optional[str] = None
    name_es: Optional[str] = None
    sku: Optional[str] = None
    qty: Optional[int] = None
    reorder_level: Optional[int] = None
    price: Optional[float] = None


# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
def root():
    if os.path.exists(INDEX_HTML):
        with open(INDEX_HTML, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Missing static/index.html</h3>"


@app.get("/health")
def health():
    return {"message": "FastAPI backend is running"}


# ---------- API ----------
@app.post("/api/login")
def login(payload: LoginIn):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND active=1", (payload.username,))
        u = cur.fetchone()

    if not u:
        raise HTTPException(status_code=401, detail="Bad credentials")

    u = dict(u)
    if not pwd_context.verify(payload.pin, u["pin_hash"]):
        raise HTTPException(status_code=401, detail="Bad credentials")

    return {"token": u["username"], "role": u["role"], "unit_id": u["unit_id"]}


@app.get("/api/me")
def me(user=Depends(require_user)):
    # Manager always treated as can_purchase=1 for display consistency
    is_manager = user["role"] == "manager"
    return {
        "username": user["username"],
        "role": user["role"],
        "unit_id": user["unit_id"],
        "can_purchase": 1 if is_manager else int(user.get("can_purchase", 0)),
    }


@app.get("/api/products")
def products(user=Depends(require_user)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM products ORDER BY category, name_es, name_en")
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["is_low"] = int(d["qty"]) <= int(d["reorder_level"])
            out.append(d)
        return out


# ---- Product management endpoints (manager-only) ----
@app.post("/api/products")
def create_product(payload: ProductCreateIn, manager=Depends(require_manager)):
    name_en = (payload.name_en or "").strip()
    name_es = (payload.name_es or "").strip()
    sku = (payload.sku or "").strip()
    category = (payload.category or "").strip()

    if not name_en:
        raise HTTPException(status_code=400, detail="name_en required")
    if not name_es:
        raise HTTPException(status_code=400, detail="name_es required")
    if not sku:
        raise HTTPException(status_code=400, detail="sku required")

    qty = int(payload.qty or 0)
    reorder_level = int(payload.reorder_level or 0)
    price = float(payload.price or 0)

    if qty < 0:
        raise HTTPException(status_code=400, detail="qty must be >= 0")
    if reorder_level < 0:
        raise HTTPException(status_code=400, detail="reorder_level must be >= 0")
    if price < 0:
        raise HTTPException(status_code=400, detail="price must be >= 0")

    now = _utc_iso_now()

    with db_conn() as conn:
        cur = conn.cursor()

        # Prevent duplicate SKU (schema doesn't enforce UNIQUE)
        cur.execute("SELECT id FROM products WHERE sku = ?", (sku,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="sku already exists")

        cur.execute("""
            INSERT INTO products (category, name_en, name_es, sku, qty, reorder_level, price, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (category, name_en, name_es, sku, qty, reorder_level, price, now))

        new_id = cur.lastrowid
        cur.execute("SELECT * FROM products WHERE id = ?", (new_id,))
        return dict(cur.fetchone())


@app.put("/api/products/{product_id}")
def update_product(product_id: int, payload: ProductUpdateIn, manager=Depends(require_manager)):
    if product_id <= 0:
        raise HTTPException(status_code=400, detail="invalid product_id")

    fields = {}
    if payload.category is not None:
        fields["category"] = payload.category.strip()
    if payload.name_en is not None:
        fields["name_en"] = payload.name_en.strip()
    if payload.name_es is not None:
        fields["name_es"] = payload.name_es.strip()
    if payload.sku is not None:
        fields["sku"] = payload.sku.strip()
    if payload.qty is not None:
        fields["qty"] = int(payload.qty)
    if payload.reorder_level is not None:
        fields["reorder_level"] = int(payload.reorder_level)
    if payload.price is not None:
        fields["price"] = float(payload.price)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "name_en" in fields and not fields["name_en"]:
        raise HTTPException(status_code=400, detail="name_en cannot be empty")
    if "name_es" in fields and not fields["name_es"]:
        raise HTTPException(status_code=400, detail="name_es cannot be empty")
    if "sku" in fields and not fields["sku"]:
        raise HTTPException(status_code=400, detail="sku cannot be empty")
    if "qty" in fields and fields["qty"] < 0:
        raise HTTPException(status_code=400, detail="qty must be >= 0")
    if "reorder_level" in fields and fields["reorder_level"] < 0:
        raise HTTPException(status_code=400, detail="reorder_level must be >= 0")
    if "price" in fields and fields["price"] < 0:
        raise HTTPException(status_code=400, detail="price must be >= 0")

    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id FROM products WHERE id = ?", (product_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="product not found")

        if "sku" in fields:
            cur.execute("SELECT id FROM products WHERE sku = ? AND id != ?", (fields["sku"], product_id))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="sku already exists")

        set_clause = ", ".join([f"{k} = ?" for k in fields.keys()])
        values = list(fields.values()) + [product_id]

        cur.execute(f"UPDATE products SET {set_clause} WHERE id = ?", values)

        cur.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        return dict(cur.fetchone())


@app.post("/api/sales")
def add_sale(payload: SaleIn, user=Depends(require_user)):
    if payload.qty <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")

    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id, qty, price FROM products WHERE id=?", (payload.product_id,))
        p = cur.fetchone()
        if not p:
            raise HTTPException(status_code=404, detail="product not found")

        p = dict(p)
        if int(p["qty"]) < int(payload.qty):
            raise HTTPException(status_code=400, detail="Insufficient inventory")

        cur.execute("UPDATE products SET qty = qty - ? WHERE id = ?", (payload.qty, payload.product_id))

        now = _utc_iso_now()
        cur.execute("""
            INSERT INTO sales (product_id, employee_id, qty, price, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (payload.product_id, user["id"], payload.qty, float(p["price"]), now))

    return {"ok": True}


@app.post("/api/purchases")
def add_purchase(payload: PurchaseLineIn, user=Depends(require_purchase_permission)):
    if payload.qty <= 0:
        raise HTTPException(status_code=400, detail="qty must be > 0")

    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT id FROM products WHERE id=?", (payload.product_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="product not found")

        cur.execute("UPDATE products SET qty = qty + ? WHERE id = ?", (payload.qty, payload.product_id))

        now = _utc_iso_now()
        cur.execute("""
            INSERT INTO purchases (product_id, employee_id, qty, cost, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (payload.product_id, user["id"], payload.qty, float(payload.cost or 0), now))

    return {"ok": True}


@app.post("/api/purchases/bulk")
def add_purchases_bulk(payload: PurchaseBulkIn, user=Depends(require_purchase_permission)):
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items provided")

    now = _utc_iso_now()

    with db_conn() as conn:
        cur = conn.cursor()

        # Validate all products first
        for it in payload.items:
            if it.qty <= 0:
                raise HTTPException(status_code=400, detail="All qty must be > 0")
            cur.execute("SELECT id FROM products WHERE id=?", (it.product_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail=f"product not found: {it.product_id}")

        # Apply updates + inserts
        for it in payload.items:
            cur.execute("UPDATE products SET qty = qty + ? WHERE id = ?", (it.qty, it.product_id))
            cur.execute("""
                INSERT INTO purchases (product_id, employee_id, qty, cost, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (it.product_id, user["id"], it.qty, float(it.cost or 0), now))

    return {"ok": True, "count": len(payload.items)}


@app.get("/api/cash")
def get_cash(manager=Depends(require_manager)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM cash ORDER BY created_at DESC LIMIT 200")
        return [dict(r) for r in cur.fetchall()]


@app.post("/api/cash")
def add_cash(payload: CashIn, manager=Depends(require_manager)):
    now = _utc_iso_now()
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO cash (amount, note, created_at) VALUES (?, ?, ?)",
            (float(payload.amount), payload.note or "", now),
        )
    return {"ok": True}


@app.get("/api/users")
def list_users(manager=Depends(require_manager)):
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, username, role, unit_id, can_purchase, active, created_at
            FROM users
            ORDER BY role, username
        """)
        return [dict(r) for r in cur.fetchall()]


@app.post("/api/users")
def create_user(payload: UserCreateIn, manager=Depends(require_manager)):
    if not payload.username.strip():
        raise HTTPException(status_code=400, detail="username required")
    if not payload.unit_id.strip():
        raise HTTPException(status_code=400, detail="unit_id required")
    if not (payload.pin.isdigit() and len(payload.pin) == 6):
        raise HTTPException(status_code=400, detail="PIN must be 6 digits")

    can_purchase = int(payload.can_purchase or 0)
    if can_purchase not in (0, 1):
        raise HTTPException(status_code=400, detail="can_purchase must be 0 or 1")

    now = _utc_iso_now()

    try:
        with db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO users (username, role, unit_id, pin_hash, active, can_purchase, created_at)
                VALUES (?, 'employee', ?, ?, 1, ?, ?)
            """, (
                payload.username.strip(),
                payload.unit_id.strip(),
                pwd_context.hash(payload.pin),
                can_purchase,
                now
            ))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="username already exists")

    return {"ok": True}


@app.post("/api/users/set-purchase-permission")
def set_purchase_permission(payload: SetPurchasePermIn, manager=Depends(require_manager)):
    if payload.can_purchase not in (0, 1):
        raise HTTPException(status_code=400, detail="can_purchase must be 0 or 1")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET can_purchase=? WHERE id=?", (payload.can_purchase, payload.user_id))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="user not found")

    return {"ok": True}


@app.post("/api/me/change-pin")
def change_pin(payload: ChangePinIn, user=Depends(require_user)):
    if not (payload.new_pin.isdigit() and len(payload.new_pin) == 6):
        raise HTTPException(status_code=400, detail="New PIN must be 6 digits")

    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pin_hash FROM users WHERE id=?", (user["id"],))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="user not found")

        current_hash = row["pin_hash"]

        if not pwd_context.verify(payload.old_pin, current_hash):
            raise HTTPException(status_code=400, detail="Old PIN incorrect")

        cur.execute(
            "UPDATE users SET pin_hash=? WHERE id=?",
            (pwd_context.hash(payload.new_pin), user["id"])
        )

    return {"ok": True}


@app.get("/api/reports/daily")
def report_daily(date: Optional[str] = None, manager=Depends(require_manager)):
    # Stored timestamps are UTC ISO strings; report uses UTC day boundaries
    if date:
        try:
            d = datetime.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    else:
        d = datetime.datetime.now(datetime.timezone.utc).date()

    start_dt = datetime.datetime.combine(d, datetime.time.min)  # 00:00:00
    next_day_dt = start_dt + datetime.timedelta(days=1)         # next day 00:00:00

    start = start_dt.isoformat()
    end = next_day_dt.isoformat()  # half-open [start, end)

    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT
              COALESCE(SUM(qty), 0) AS items_sold,
              COUNT(*) AS transactions,
              COALESCE(SUM(qty * COALESCE(price, 0)), 0) AS revenue
            FROM sales
            WHERE created_at >= ? AND created_at < ?
        """, (start, end))
        totals = dict(cur.fetchone())

        cur.execute("""
            SELECT
              u.username AS employee,
              COALESCE(u.unit_id,'') AS unit_id,
              COALESCE(SUM(s.qty), 0) AS items_sold,
              COUNT(*) AS transactions,
              COALESCE(SUM(s.qty * COALESCE(s.price, 0)), 0) AS revenue
            FROM sales s
            JOIN users u ON u.id = s.employee_id
            WHERE s.created_at >= ? AND s.created_at < ?
            GROUP BY u.username, u.unit_id
            ORDER BY revenue DESC
        """, (start, end))
        per_employee = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT id, name_en, name_es, qty, reorder_level
            FROM products
            WHERE qty <= reorder_level
            ORDER BY qty ASC
        """)
        low_stock = [dict(r) for r in cur.fetchall()]

    return {"date": d.isoformat(), "totals": totals, "per_employee": per_employee, "low_stock": low_stock}


@app.get("/api/purchases/history")
def purchases_history(limit: int = 200, employee: Optional[str] = None, user=Depends(require_user)):
    limit = max(1, min(int(limit), 500))

    with db_conn() as conn:
        cur = conn.cursor()

        # Manager: can view all or filter by employee username
        if user["role"] == "manager":
            params = []
            where = "1=1"
            if employee:
                where = "u.username = ?"
                params.append(employee.strip())

            cur.execute(f"""
                SELECT
                  p.id,
                  p.created_at,
                  p.qty,
                  p.cost,
                  pr.name_es,
                  pr.name_en,
                  pr.sku,
                  u.username AS employee,
                  COALESCE(u.unit_id,'') AS unit_id
                FROM purchases p
                JOIN products pr ON pr.id = p.product_id
                LEFT JOIN users u ON u.id = p.employee_id
                WHERE {where}
                ORDER BY p.created_at DESC
                LIMIT ?
            """, (*params, limit))
            return [dict(r) for r in cur.fetchall()]

        # Employee: only view own purchase history
        cur.execute("""
            SELECT
              p.id,
              p.created_at,
              p.qty,
              p.cost,
              pr.name_es,
              pr.name_en,
              pr.sku,
              ? AS employee,
              ? AS unit_id
            FROM purchases p
            JOIN products pr ON pr.id = p.product_id
            WHERE p.employee_id = ?
            ORDER BY p.created_at DESC
            LIMIT ?
        """, (user["username"], user.get("unit_id", ""), user["id"], limit))
        return [dict(r) for r in cur.fetchall()]
