# Frontend Engineering Standards & Evaluation Criteria
## Mandatory Compliance Directives for UI Development Personnel

---

## Instruction 1: Architecture & Codebase Discipline

Structure the frontend codebase utilizing strictly modular, domain-driven architecture patterns. Enforce unequivocal separation across the following layers:

- **Presentation Layer**: Pure render logic, zero side-effects
- **State Orchestration Layer**: Centralized state containers, reducers, selectors
- **Side-Effects Layer**: Async operations, API integrations, subscriptions
- **Infrastructure Layer**: Platform bindings, environment configurations, adapters

**Mandatory Constraints:**
- Prohibit all cross-layer import statements; enforce via linting rules and architectural boundary tests
- Guarantee each module achieves independent testability without mocking adjacent layers
- Ensure all exports satisfy tree-shaking compatibility; avoid barrel files that circumvent dead-code elimination
- Implement feature-sliced directory topology with explicit public API surfaces per module
- Maintain dependency inversion at layer boundaries via interface contracts

---

## Instruction 2: Type System Absolutism

Configure TypeScript compiler with maximum strictness parameters:

```
strict: true
noUncheckedIndexedAccess: true
exactOptionalPropertyTypes: true
noImplicitReturns: true
noFallthroughCasesInSwitch: true
noPropertyAccessFromIndexSignature: true
```

**Zero-Tolerance Violations:**
- Usage of `any` type under any circumstance
- Type assertions (`as` keyword) without accompanying runtime validation
- Implicit inference leakage through untyped function parameters or return values
- Non-validated external data ingestion from APIs, localStorage, URL parameters, or user inputs

**Mandatory Implementation:**
- Deploy schema-driven validation libraries (Zod, io-ts, Valibot) at all external data boundaries
- Generate API client types from OpenAPI/GraphQL schemas; prohibit manual type duplication
- Enforce branded types for domain primitives requiring semantic distinction (UserId, Timestamp, Currency)
- Utilize discriminated unions for exhaustive pattern matching on variant types

---

## Instruction 3: Rendering & Performance Determinism

Guarantee deterministic rendering behavior across all execution contexts:

**Concurrent Rendering Compliance:**
- Ensure all components remain idempotent under React concurrent features (useTransition, useDeferredValue)
- Eliminate render-phase side-effects; restrict effects exclusively to useEffect, useLayoutEffect boundaries
- Handle Suspense boundaries with explicit fallback hierarchies and error boundary coverage

**Re-render Elimination Protocol:**
- Apply memoization (React.memo, useMemo, useCallback) exclusively based on empirical profiling data
- Prohibit speculative memoization without measured performance degradation evidence
- Implement stable reference patterns for callback props and context values
- Utilize selector functions with memoized derivation for store subscriptions

**Performance Budget Enforcement:**
- Define quantitative thresholds: Largest Contentful Paint < 2.5s, First Input Delay < 100ms, Cumulative Layout Shift < 0.1
- Integrate performance assertions into continuous integration pipeline
- Profile rendering flame graphs on representative device hardware; prohibit development-only benchmarking

---

## Instruction 4: State Management Formalism

Model application state utilizing formal state management paradigms:

**State Machine Implementation:**
- Represent complex interaction flows via explicit finite state machines (XState, Stately)
- Define all possible states, transitions, and guards declaratively
- Prohibit implicit boolean flag combinations representing pseudo-states

**Event-Driven Store Architecture:**
- Dispatch typed action events representing user intent or system occurrences
- Process state mutations through pure reducer functions with deterministic outputs
- Isolate asynchronous operations into middleware layers (sagas, thunks, observables) with explicit effect tracking

**Component Statelessness Mandate:**
- Restrict UI components to pure render functions consuming props and derived state
- Elevate all persistent state to appropriate store granularity (local, feature, global)
- Permit ephemeral UI state (hover, focus) within components; prohibit business logic coupling

**Traceability Requirements:**
- Enable complete state transition logging in development environments
- Implement time-travel debugging capability for state history inspection
- Ensure all state mutations trace to originating user actions or system events

---

## Instruction 5: Design System Governance

Construct token-driven design system architecture with immutable primitive foundations:

**Token Taxonomy:**
- **Spacing Scale**: Defined mathematical progression (4px base unit, geometric/modular scale)
- **Typography Scale**: Font families, weights, sizes, line-heights, letter-spacing as discrete tokens
- **Color Palettes**: Semantic color assignments (foreground, background, accent, destructive) mapped to primitive values
- **Motion Tokens**: Duration, easing curves, stagger offsets as reusable animation primitives
- **Elevation Tokens**: Shadow definitions, z-index stratification, layering semantics

**Enforcement Mechanisms:**
- Prohibit hardcoded CSS values (px, rem, hex, rgb) outside token definitions
- Implement stylelint/eslint rules detecting ad-hoc value usage
- Generate platform-agnostic token distributions (CSS custom properties, JSON, TypeScript constants)
- Version control design token repository with semantic versioning and changelog documentation

**Component API Standardization:**
- Define exhaustive prop interfaces for all primitive components
- Restrict styling customization to token-based variant props; prohibit className/style prop exposure on primitives
- Document component usage patterns with interactive examples in isolated component workshop environment

---

## Instruction 6: Accessibility as Hard Constraint

Achieve WCAG 2.2 Level AAA compliance through construction-phase integration:

**Semantic Structure Requirements:**
- Utilize appropriate HTML5 sectioning elements (main, nav, article, aside, header, footer)
- Implement heading hierarchy (h1-h6) reflecting document outline without skipping levels
- Associate form inputs with labels programmatically via htmlFor/id binding or implicit nesting

**Keyboard Navigation Mandate:**
- Guarantee all interactive elements receive focus via Tab navigation in logical order
- Implement visible focus indicators meeting 3:1 contrast ratio minimum
- Provide keyboard equivalents (Enter, Space, Escape, Arrow keys) for all pointer-based interactions
- Manage focus programmatically during modal dialogs, route transitions, and dynamic content updates

**ARIA Implementation Protocol:**
- Apply ARIA roles, states, and properties only when native HTML semantics prove insufficient
- Validate ARIA usage against WAI-ARIA Authoring Practices specifications
- Implement live regions (aria-live, aria-atomic) for dynamic content announcements
- Test with multiple screen readers (NVDA, JAWS, VoiceOver) across target platforms

**Validation Integration:**
- Execute automated accessibility audits (axe-core) as build-blocking quality gates
- Conduct manual accessibility testing with assistive technology users
- Classify accessibility violations equivalent to functional defects; prohibit release with outstanding violations

---

## Instruction 7: Visual Precision & Interaction Fidelity

Deliver pixel-accurate implementations with mathematically rigorous alignment:

**Layout Precision Standards:**
- Implement designs with sub-pixel rendering awareness; utilize transform: translateZ(0) for compositor layer promotion where applicable
- Align elements to pixel grid boundaries to prevent anti-aliasing artifacts on low-DPI displays
- Calculate responsive breakpoints and fluid typography using precise mathematical formulas (clamp, min, max, calc)
- Validate layout accuracy against design specifications using automated visual comparison tools with configurable tolerance thresholds

**Animation & Motion Requirements:**
- Define micro-interactions using physically coherent motion models (spring physics, easing functions derived from natural motion)
- Maintain frame stability at 60 FPS minimum; target 120 FPS on high-refresh-rate displays
- Utilize compositor-accelerated properties exclusively (transform, opacity) for animations
- Implement reduced-motion media query alternatives respecting user accessibility preferences

**Rendering Performance Constraints:**
- Avoid layout thrashing; batch DOM read/write operations
- Utilize CSS containment (contain: layout, paint, size) to isolate rendering subtrees
- Implement virtualization for unbounded list rendering with stable scroll position maintenance
- Profile paint and composite operations; eliminate unnecessary layer creation

---

## Instruction 8: Testing & Verification Rigor

Implement multi-layer testing strategy with comprehensive coverage:

**Unit Testing Protocol:**
- Test pure functions, hooks, and utilities in isolation with deterministic inputs/outputs
- Mock external dependencies at module boundaries; prohibit deep implementation mocking
- Achieve branch coverage thresholds appropriate to module criticality (minimum 80% for core logic)

**Integration Testing Requirements:**
- Validate component composition behavior through user-centric interaction simulation
- Utilize Testing Library philosophy: query by accessible semantics, not implementation details
- Test asynchronous flows with explicit wait conditions; prohibit arbitrary timeout usage

**Visual Regression Testing:**
- Capture component snapshots across variant permutations and viewport dimensions
- Integrate screenshot comparison into pull request workflow with diff highlighting
- Maintain baseline images under version control with documented update procedures

**Accessibility Audit Integration:**
- Execute axe-core assertions within component test suites
- Validate keyboard navigation flows programmatically
- Test focus management during dynamic interactions

**Test Quality Standards:**
- Assert observable behavior, not internal implementation state
- Eliminate flaky tests immediately upon detection; investigate root cause systematically
- Prohibit test-specific conditional logic in production code
- Maintain test isolation; no shared mutable state between test cases

---

## Instruction 9: Build, Tooling & Observability

Establish deterministic build infrastructure with comprehensive observability:

**Build Pipeline Requirements:**
- Guarantee reproducible builds via lockfile enforcement and pinned dependency versions
- Implement content-addressable caching for build artifacts
- Execute parallelized build steps with dependency graph optimization

**Bundle Optimization Mandate:**
- Analyze bundle composition continuously; monitor for unexpected size regressions
- Enforce code-splitting at route boundaries and dynamic import points
- Eliminate dead code through tree-shaking verification; audit side-effect declarations
- Configure differential serving (modern/legacy bundles) based on browser capability detection

**Dependency Governance:**
- Audit transitive dependencies for security vulnerabilities and license compliance
- Prohibit unnecessary dependencies; justify each addition against bundle impact
- Implement automated dependency update workflows with breaking change detection

**Runtime Observability Integration:**
- Instrument rendering latency metrics (Time to Interactive, component mount duration)
- Track interaction timing (click-to-response, input latency)
- Implement error boundary telemetry with stack trace aggregation and session replay capability
- Export metrics to observability platform with appropriate sampling rates and cardinality management

**Telemetry Standards:**
- Define custom performance marks/measures for critical user journeys
- Correlate frontend metrics with backend distributed traces
- Establish alerting thresholds for performance degradation detection

---

## Instruction 10: Engineering Ethics & Technical Judgment

Exercise exceptional technical discernment in all engineering decisions:

**Simplicity Imperative:**
- Prefer straightforward implementations over clever abstractions requiring cognitive overhead
- Evaluate solutions against comprehensibility by team members with median domain familiarity
- Prohibit premature optimization without quantified performance justification

**Explicitness Standard:**
- Favor explicit configuration over convention-based magic
- Document non-obvious design decisions at point of implementation
- Name entities (variables, functions, components) descriptively reflecting purpose, not mechanism

**Maintainability Prioritization:**
- Evaluate architectural decisions against five-year maintenance horizon
- Prefer boring, proven technologies over novel solutions lacking production validation
- Design for deletion: ensure features can be removed without widespread codebase modification

**Abstraction Justification Protocol:**
- Require each abstraction layer demonstrate measurable value (code reduction, consistency enforcement, complexity encapsulation)
- Prohibit abstractions created speculatively for anticipated future requirements
- Document abstraction boundaries with explicit interface contracts and usage constraints

**Technical Integrity:**
- Acknowledge technical debt explicitly; maintain living documentation of known deficiencies
- Communicate tradeoffs transparently during architectural discussions
- Refuse to compromise on correctness for schedule pressure; escalate resource constraints appropriately

---

## Compliance Verification

All frontend engineering personnel shall demonstrate adherence to these directives through:
- Code review validation against enumerated standards
- Automated linting and testing enforcement in continuous integration
- Periodic architectural review sessions evaluating systemic compliance
- Performance and accessibility audits conducted on release cadence

Non-compliance with any directive requires documented justification approved by technical leadership.

---

**Document Classification**: Internal Engineering Standards
**Revision Authority**: Frontend Architecture Council
**Effective Immediately Upon Distribution**
------------------------------------
# Backend Engineering Standards & Evaluation Criteria
## Mandatory Compliance Directives for API Development Personnel

---

## Instruction 1: Architecture & Codebase Discipline

Structure the backend codebase utilizing strictly layered, domain-driven architecture patterns. Enforce unequivocal separation across the following layers:

- **Presentation Layer**: Request/response handling, serialization, validation
- **Application Layer**: Use cases, orchestration, transaction boundaries
- **Domain Layer**: Business logic, entities, domain services, aggregates
- **Infrastructure Layer**: Database adapters, external service clients, messaging

**Mandatory Constraints:**
- Prohibit direct database access from presentation layer; enforce repository pattern abstraction
- Implement dependency injection for all cross-layer dependencies; prohibit hard-coded instantiation
- Guarantee each module achieves independent testability via interface segregation
- Maintain single responsibility principle per module; maximum cyclomatic complexity threshold of 10 per function
- Enforce hexagonal/ports-and-adapters pattern for external system integrations

```
src/
├── api/              # Presentation: routes, controllers, schemas
├── application/      # Use cases, DTOs, application services
├── domain/           # Entities, value objects, domain services
├── infrastructure/   # Repositories, external clients, adapters
└── core/             # Shared utilities, exceptions, constants
```

---

## Instruction 2: RESTful API Design & HTTP Semantics Compliance

Implement API endpoints adhering to strict REST architectural constraints and HTTP specification compliance:

**Resource Naming Standards:**
- Utilize plural nouns for collection endpoints (`/users`, `/orders`, `/products`)
- Implement hierarchical nesting for sub-resources (`/users/{user_id}/orders/{order_id}`)
- Prohibit verbs in URI paths; actions derive from HTTP methods exclusively
- Apply kebab-case for multi-word resource identifiers (`/order-items`, `/payment-methods`)

**HTTP Method Semantics:**
```
GET     → Retrieve resource(s); idempotent, cacheable, safe
POST    → Create resource; non-idempotent, returns 201 with Location header
PUT     → Full resource replacement; idempotent, returns 200/204
PATCH   → Partial resource modification; idempotent, returns 200
DELETE  → Resource removal; idempotent, returns 204/202
OPTIONS → CORS preflight, capability discovery
HEAD    → Metadata retrieval without response body
```

**Status Code Precision:**
```
200 OK                    → Successful retrieval/modification
201 Created               → Resource creation with Location header
202 Accepted              → Async operation initiated
204 No Content            → Successful operation, empty response body
400 Bad Request           → Malformed request syntax, validation failure
401 Unauthorized          → Missing/invalid authentication credentials
403 Forbidden             → Valid credentials, insufficient permissions
404 Not Found             → Resource does not exist
405 Method Not Allowed    → HTTP method unsupported for resource
409 Conflict              → Resource state conflict (optimistic locking)
422 Unprocessable Entity  → Semantic validation failure
429 Too Many Requests     → Rate limit exceeded with Retry-After header
500 Internal Server Error → Unhandled server-side exception
502 Bad Gateway           → Upstream service failure
503 Service Unavailable   → Temporary overload with Retry-After header
504 Gateway Timeout       → Upstream service timeout
```

**Prohibit status code misuse; each response must semantically align with specification.**

---

## Instruction 3: CRUD Operations & Database Transaction Management

Implement data persistence operations with transactional integrity and optimized query execution:

**Repository Pattern Implementation:**
```python
class BaseRepository(Generic[T]):
    async def create(self, entity: T) -> T
    async def get_by_id(self, id: UUID) -> T | None
    async def get_all(self, filters: FilterParams) -> list[T]
    async def update(self, id: UUID, data: UpdateSchema) -> T
    async def delete(self, id: UUID) -> None
    async def exists(self, id: UUID) -> bool
    async def count(self, filters: FilterParams) -> int
    async def bulk_create(self, entities: list[T]) -> list[T]
    async def bulk_update(self, updates: list[BulkUpdate]) -> int
    async def bulk_delete(self, ids: list[UUID]) -> int
```

**Transaction Boundary Management:**
- Demarcate transaction boundaries at application/use-case layer exclusively
- Implement unit-of-work pattern for multi-repository operations
- Configure appropriate isolation levels per operation criticality:
  - `READ COMMITTED`: Standard read operations
  - `REPEATABLE READ`: Financial calculations, inventory checks
  - `SERIALIZABLE`: Critical consistency requirements
- Handle deadlock detection with exponential backoff retry strategy

**Optimistic Concurrency Control:**
- Implement version columns for concurrent modification detection
- Return 409 Conflict on version mismatch with current state representation
- Prohibit pessimistic locking except for explicitly justified scenarios

**Soft Delete Implementation:**
- Implement `deleted_at` timestamp for recoverable deletion
- Apply global query filters excluding soft-deleted records by default
- Expose explicit endpoints for permanent deletion with authorization checks

---

## Instruction 4: Query Parameters & Filtering Architecture

Implement comprehensive query parameter handling with type-safe parsing and validation:

**Pagination Standards:**
```
GET /resources?page=1&page_size=20      # Offset-based pagination
GET /resources?cursor=eyJpZCI6MTAwfQ    # Cursor-based pagination (preferred)
GET /resources?limit=20&offset=40       # Explicit offset (legacy support)
```

**Response Envelope Structure:**
```json
{
  "data": [...],
  "meta": {
    "total_count": 1000,
    "page": 1,
    "page_size": 20,
    "total_pages": 50,
    "has_next": true,
    "has_previous": false,
    "next_cursor": "eyJpZCI6MTIwfQ",
    "previous_cursor": null
  },
  "links": {
    "self": "/resources?page=1",
    "next": "/resources?page=2",
    "previous": null,
    "first": "/resources?page=1",
    "last": "/resources?page=50"
  }
}
```

**Filtering Specification:**
```
GET /resources?status=active                           # Exact match
GET /resources?created_at[gte]=2024-01-01             # Range operators
GET /resources?name[contains]=search                   # Partial match
GET /resources?tags[in]=tag1,tag2,tag3                # Set membership
GET /resources?price[between]=100,500                  # Range boundaries
GET /resources?email[is_null]=false                    # Null checks
```

**Sorting Implementation:**
```
GET /resources?sort=created_at                         # Ascending (default)
GET /resources?sort=-created_at                        # Descending (prefix -)
GET /resources?sort=status,-created_at                 # Multi-column sorting
```

**Field Selection (Sparse Fieldsets):**
```
GET /resources?fields=id,name,status                   # Explicit field inclusion
GET /resources?include=author,comments                 # Relationship expansion
```

**Query Validation Requirements:**
- Validate all query parameters against defined schema; reject unknown parameters
- Enforce maximum page_size limits (configurable, default 100)
- Sanitize filter values preventing SQL injection via parameterized queries
- Index database columns utilized in filter/sort operations

---

## Instruction 5: Path Parameters & Route Resolution

Implement path parameter handling with strict validation and hierarchical resource resolution:

**Path Parameter Typing:**
```python
@router.get("/users/{user_id}")
async def get_user(
    user_id: Annotated[UUID, Path(description="Unique user identifier")]
) -> UserResponse:
    ...

@router.get("/orders/{order_id}/items/{item_id}")
async def get_order_item(
    order_id: Annotated[UUID, Path(...)],
    item_id: Annotated[UUID, Path(...)]
) -> OrderItemResponse:
    ...
```

**Hierarchical Resource Validation:**
- Validate parent resource existence before child resource operations
- Return 404 if parent resource does not exist (not 403 to prevent enumeration)
- Verify relationship integrity between hierarchical path components

**Path Parameter Constraints:**
```python
# UUID validation
user_id: Annotated[UUID, Path(regex="^[0-9a-f]{8}-...")]

# Slug validation
slug: Annotated[str, Path(min_length=3, max_length=50, regex="^[a-z0-9-]+$")]

# Numeric ID validation
legacy_id: Annotated[int, Path(gt=0, lt=2147483647)]

# Enum constraint
status: Annotated[OrderStatus, Path(...)]
```

**Route Registration Order:**
- Register specific routes before parameterized routes to prevent shadowing
- Implement explicit route conflict detection during application startup
- Document route precedence rules in routing configuration

**Canonical URL Enforcement:**
- Redirect non-canonical URLs (trailing slashes, case variations) to canonical form
- Return 301 Permanent Redirect for SEO-sensitive endpoints
- Maintain consistent URL structure across API versions

---

## Instruction 6: CORS Configuration & Security Headers

Implement Cross-Origin Resource Sharing with principle of least privilege:

**CORS Configuration Parameters:**
```python
CORSMiddleware(
    allow_origins=["https://app.example.com", "https://admin.example.com"],
    allow_origin_regex=r"https://.*\.example\.com",
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Correlation-ID"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    allow_credentials=True,
    max_age=86400,  # Preflight cache duration
)
```

**CORS Security Mandates:**
- Prohibit wildcard (`*`) origins in production environments when credentials enabled
- Enumerate allowed origins explicitly from environment configuration
- Validate Origin header against allowlist; reject unmatched origins silently
- Implement per-route CORS overrides for endpoints requiring different policies

**Preflight Request Handling:**
- Respond to OPTIONS requests with appropriate Access-Control-* headers
- Cache preflight responses via Access-Control-Max-Age header
- Return 204 No Content for successful preflight responses

**Security Header Enforcement:**
```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 0  # Deprecated, rely on CSP
Content-Security-Policy: default-src 'self'; frame-ancestors 'none'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Cache-Control: no-store, no-cache, must-revalidate (for sensitive endpoints)
```

**Security Validation:**
- Audit CORS configuration during deployment pipeline
- Test preflight behavior across all HTTP methods
- Verify credentials handling with cross-origin requests

---

## Instruction 7: Background Task & Async Job Processing

Implement background task orchestration with reliability, observability, and failure resilience:

**Task Queue Architecture:**
```python
# Task definition with explicit configuration
@task_queue.task(
    name="process_order",
    queue="high_priority",
    max_retries=3,
    retry_backoff=ExponentialBackoff(base=60, max=3600),
    timeout=300,
    acks_late=True,
    reject_on_worker_lost=True,
)
async def process_order(order_id: UUID, correlation_id: str) -> TaskResult:
    ...
```

**Task Classification & Queue Segregation:**
```
critical_queue    → Payment processing, order fulfillment (dedicated workers)
high_priority     → Email notifications, webhook deliveries
default_queue     → Report generation, data synchronization
low_priority      → Analytics aggregation, cleanup operations
scheduled_queue   → Cron-based periodic tasks
```

**Idempotency Implementation:**
- Generate unique idempotency keys for all task invocations
- Implement deduplication via distributed locking or database constraints
- Design task handlers to produce identical outcomes on repeated execution
- Store task execution results for idempotency verification

**Failure Handling Protocol:**
```python
class TaskRetryPolicy:
    max_retries: int = 3
    retry_delays: list[int] = [60, 300, 900]  # Progressive backoff
    retry_on: tuple[type[Exception], ...] = (TransientError, TimeoutError)
    fail_on: tuple[type[Exception], ...] = (ValidationError, AuthorizationError)
    dead_letter_queue: str = "failed_tasks"
```

**Task Observability Requirements:**
- Emit structured logs at task start, completion, and failure
- Track task duration, queue wait time, and retry counts as metrics
- Propagate correlation IDs from originating requests through task chains
- Implement dead-letter queue processing with alerting integration

**Scheduled Task Management:**
- Define cron schedules declaratively with timezone awareness
- Implement distributed locking preventing concurrent schedule execution
- Monitor schedule drift and execution latency
- Provide manual trigger capability for operational recovery

---

## Instruction 8: Request/Response Validation & Error Handling

Implement comprehensive validation with structured error responses:

**Request Validation Schema:**
```python
class CreateOrderRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    
    customer_id: Annotated[UUID, Field(description="Customer identifier")]
    items: Annotated[list[OrderItem], Field(min_length=1, max_length=100)]
    shipping_address: Address
    payment_method_id: UUID
    notes: Annotated[str | None, Field(max_length=500)] = None
    
    @field_validator("items")
    @classmethod
    def validate_unique_products(cls, items: list[OrderItem]) -> list[OrderItem]:
        product_ids = [item.product_id for item in items]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Duplicate product IDs not permitted")
        return items
```

**Standardized Error Response Format:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "items[0].quantity",
        "code": "VALUE_OUT_OF_RANGE",
        "message": "Quantity must be between 1 and 1000",
        "context": {"min": 1, "max": 1000, "actual": 0}
      }
    ],
    "request_id": "req_abc123",
    "timestamp": "2024-01-15T10:30:00Z",
    "documentation_url": "https://api.example.com/docs/errors#VALIDATION_ERROR"
  }
}
```

**Error Code Taxonomy:**
```
VALIDATION_ERROR          → 400: Request body/parameter validation failure
AUTHENTICATION_REQUIRED   → 401: Missing or invalid authentication
PERMISSION_DENIED         → 403: Insufficient authorization
RESOURCE_NOT_FOUND        → 404: Requested resource does not exist
METHOD_NOT_ALLOWED        → 405: HTTP method not supported
RESOURCE_CONFLICT         → 409: State conflict (duplicate, version mismatch)
RATE_LIMIT_EXCEEDED       → 429: Request throttling active
INTERNAL_ERROR            → 500: Unhandled server exception
SERVICE_UNAVAILABLE       → 503: Dependency failure, maintenance mode
```

**Exception Handling Hierarchy:**
```python
class ApplicationException(Exception):
    status_code: int
    error_code: str
    message: str
    details: list[ErrorDetail] | None

class ValidationException(ApplicationException):
    status_code = 400
    error_code = "VALIDATION_ERROR"

class ResourceNotFoundException(ApplicationException):
    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"

class BusinessRuleViolation(ApplicationException):
    status_code = 422
    error_code = "BUSINESS_RULE_VIOLATION"
```

**Sensitive Data Protection:**
- Exclude stack traces from production error responses
- Sanitize error messages removing internal system references
- Log complete exception context server-side with request correlation

---

## Instruction 9: Performance Optimization & Caching Strategy

Implement performance optimizations with measurable impact validation:

**Database Query Optimization:**
```python
# N+1 query prevention via eager loading
query = (
    select(Order)
    .options(
        selectinload(Order.items).selectinload(OrderItem.product),
        joinedload(Order.customer),
    )
    .where(Order.status == OrderStatus.PENDING)
)

# Pagination with keyset/cursor approach
query = (
    select(Order)
    .where(Order.created_at < cursor_timestamp)
    .order_by(Order.created_at.desc())
    .limit(page_size + 1)  # Fetch n+1 for has_next detection
)
```

**Query Performance Mandates:**
- Analyze query execution plans for operations exceeding 100ms threshold
- Implement database connection pooling with appropriate sizing
- Prohibit unbounded queries; enforce pagination on all collection endpoints
- Index foreign keys, filter columns, and sort columns
- Monitor slow query logs with automated alerting

**Caching Layer Implementation:**
```python
@cache(
    key_builder=lambda user_id: f"user:{user_id}:profile",
    ttl=timedelta(minutes=15),
    stale_ttl=timedelta(hours=1),  # Serve stale while revalidating
    invalidation_events=["user.updated", "user.deleted"],
)
async def get_user_profile(user_id: UUID) -> UserProfile:
    ...
```

**Cache Invalidation Strategy:**
- Implement event-driven cache invalidation for write operations
- Use cache-aside pattern for read-heavy workloads
- Apply write-through pattern for consistency-critical data
- Configure appropriate TTLs based on data volatility analysis

**HTTP Caching Headers:**
```
Cache-Control: public, max-age=3600, s-maxage=86400, stale-while-revalidate=300
ETag: "a1b2c3d4e5f6"
Last-Modified: Mon, 15 Jan 2024 10:30:00 GMT
Vary: Accept, Accept-Encoding, Authorization
```

**Connection Management:**
- Configure HTTP keep-alive for persistent connections
- Implement connection pooling for database and external services
- Set appropriate timeouts (connect, read, write) for all network operations
- Monitor connection pool utilization and saturation

---

## Instruction 10: Testing, Observability & Operational Excellence

Implement comprehensive testing strategy with production-grade observability:

**Testing Pyramid Implementation:**

```python
# Unit Test: Isolated business logic
async def test_order_total_calculation():
    order = Order(items=[
        OrderItem(product_id=uuid4(), quantity=2, unit_price=Decimal("10.00")),
        OrderItem(product_id=uuid4(), quantity=1, unit_price=Decimal("25.00")),
    ])
    assert order.calculate_total() == Decimal("45.00")

# Integration Test: API endpoint with database
async def test_create_order_endpoint(client: AsyncClient, db_session: AsyncSession):
    customer = await create_test_customer(db_session)
    product = await create_test_product(db_session)
    
    response = await client.post("/orders", json={
        "customer_id": str(customer.id),
        "items": [{"product_id": str(product.id), "quantity": 2}],
    })
    
    assert response.status_code == 201
    assert response.headers["Location"].startswith("/orders/")

# Contract Test: API schema validation
async def test_order_response_schema_compliance():
    response = await client.get(f"/orders/{order_id}")
    validate_schema(response.json(), OrderResponseSchema)
```

**Test Data Management:**
- Implement factory patterns for test entity generation
- Isolate test databases with transaction rollback per test
- Prohibit production data usage in test environments
- Seed deterministic test data for reproducible scenarios

**Structured Logging Standard:**
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "message": "Order created successfully",
  "service": "order-service",
  "version": "1.2.3",
  "request_id": "req_abc123",
  "correlation_id": "corr_xyz789",
  "user_id": "user_123",
  "order_id": "order_456",
  "duration_ms": 45,
  "span_id": "span_abc",
  "trace_id": "trace_xyz"
}
```

**Metrics Exposition:**
```
http_requests_total{method, endpoint, status_code}
http_request_duration_seconds{method, endpoint}
database_query_duration_seconds{operation, table}
background_task_duration_seconds{task_name, status}
cache_hits_total{cache_name}
cache_misses_total{cache_name}
active_database_connections{pool_name}
```

**Health Check Implementation:**
```python
@router.get("/health/live")     # Kubernetes liveness probe
async def liveness() -> dict:
    return {"status": "alive"}

@router.get("/health/ready")    # Kubernetes readiness probe
async def readiness() -> dict:
    checks = {
        "database": await check_database_connection(),
        "cache": await check_cache_connection(),
        "queue": await check_queue_connection(),
    }
    status = "ready" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}

@router.get("/health/startup")  # Kubernetes startup probe
async def startup() -> dict:
    return {"status": "started", "migrations": "applied"}
```

**Distributed Tracing:**
- Propagate trace context (W3C Trace Context) across service boundaries
- Instrument database queries, HTTP clients, and message consumers
- Configure appropriate sampling rates for production traffic
- Correlate traces with logs via shared trace_id field

---

## Compliance Verification

All backend engineering personnel shall demonstrate adherence to these directives through:

- Code review validation against enumerated standards
- Automated linting, type checking, and test execution in continuous integration
- API contract testing against OpenAPI specifications
- Performance benchmarking under simulated production load
- Security scanning for dependency vulnerabilities and code patterns

Non-compliance with any directive requires documented justification approved by technical leadership.

---

**Document Classification**: Internal Engineering Standards  
**Revision Authority**: Backend Architecture Council  
**Effective Immediately Upon Distribution**