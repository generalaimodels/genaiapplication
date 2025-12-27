# FRONTEND ENGINEERING STANDARDS & PROTOCOLS

1.  **Domain-Driven Architectural Enforcement**
    Implement a strictly decoupled, module-federated architecture rooted in Domain-Driven Design (DDD) principles. Enforce hard boundaries between the Presentation Layer, Application Layer (Orchestration), Domain Layer (Business Logic), and Infrastructure Layer. Prohibit cross-domain coupling and circular dependencies; adherence must be validated via static analysis tools (e.g., Dependency Cruiser) within the CI pipeline. Ensure all modules expose explicitly defined public APIs while encapsulating internal implementation details to guarantee independent testability and efficient tree-shaking algorithms.

2.  **Total Type Safety & Algebraic Data Modeling**
    Configure TypeScript to the strictest compiler settings (`strict`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`) with zero tolerance for `any`, `unknown` casting, or non-null assertions. Model complex domain logic using Algebraic Data Types (ADTs) and Discriminated Unions to represent all possible application states explicitly. Enforce runtime integrity at the I/O boundaries by validating all external payloads (API responses, URL parameters) against rigorous schemas (e.g., Zod, io-ts) ensuring complete type narrowing and covariance/contravariance correctness.

3.  **Deterministic Rendering & Critical Path Optimization**
    Architect rendering logic to support concurrent execution contexts, leveraging techniques such as time-slicing and Suspense boundaries for non-blocking UI updates. Eliminate render-thrashing by strictly memoizing referential identities (via `useMemo`, `useCallback` or framework equivalents) based on empirical flame-graph profiling, not premature optimization. Enforce hard performance budgets on Core Web Vitals (LCP, CLS, INP); explicitly manage component lifecycles to prevent memory leaks and ensure efficient reconciliation during high-frequency state mutations.

4.  **Finite State Machine (FSM) Governance**
    Abjure boolean-heavy state flags in favor of formal Finite State Machines or Statecharts (e.g., XState) to mathematically guarantee that the UI cannot enter impossible states. Implement a unidirectional data flow pattern where UI components function as pure, deterministic projections of the application state. Isolate all side-effects (asynchronous data fetching, subscriptions) into dedicated middleware or effect layers, ensuring they are traceable, cancelable, and decoupled from the view logic.

5.  **Atomic Design System & Token Architecture**
    Construct the UI layer using an Atomic Design methodology powered by a centralized Design Token engine. Hardcode all visual primitives—typography scales, spacing, chromatic values, z-indices, and motion curves—into immutable dictionaries or CSS variables. Prohibit magic numbers and ad-hoc CSS property definitions; all style declarations must map directly to the token taxonomy to ensure themability and consistency across the component hierarchy.

6.  **Semantic Rigor & WCAG 2.2 AAA Compliance**
    Treat accessibility as a fundamental structural constraint rather than a post-development audit. Enforce semantic HTML5 usage, strictly managing the Accessibility Tree, ARIA states/properties, and focus trapping mechanisms for modals/popovers. Automated auditing tools (e.g., axe-core) must serve as build-blocking gates, complemented by manual validation of keyboard navigation flows and screen reader verbosity to achieve full WCAG 2.2 AAA conformance.

7.  **High-Fidelity Rendering & GPU Acceleration**
    Execute pixel-perfect interface implementation with sub-pixel antialiasing awareness and mathematically consistent alignment. Restrict animation properties to compositor-only layers (`transform`, `opacity`) to force hardware acceleration and prevent main-thread layout thrashing. Maintain a sustained frame rate of 60–120 FPS by utilizing `requestAnimationFrame` for motion loops and strictly managing the browser’s repaint/reflow cycles.

8.  **Pyramidic Testing Strategy & Behavioral Assertion**
    Implement a comprehensive testing pyramid comprising Static Analysis, Unit Tests, Integration Tests, and E2E scenarios. Tests must assert public behavioral contracts rather than internal implementation details to reduce brittleness. Mandate 100% coverage on critical business logic and visual regression testing for UI components. Non-deterministic (flaky) tests must be immediately isolated and remediated; mock boundaries must be typed and synchronized with actual API schemas.

9.  **Deterministic Build Pipelines & Observability**
    Engineer a reproducible, hermetic build process utilizing artifact hashing and content-addressable storage. Enforce aggressive bundle optimization strategies, including code-splitting, dead-code elimination (DCE), and granular dependency analysis. Instrument the application with real-time telemetry (RUM) to capture high-resolution metrics on rendering latency, interaction time-to-next-frame, and unhandled exceptions, enabling precise production debugging.

10. **Algorithmic Efficiency & Cognitive Complexity Management**
    Prioritize code maintainability and algorithmic efficiency (Big O notation awareness) in frontend logic. Enforce strict cyclomatic complexity thresholds and cognitive load limits via linter rules. Justify every abstraction; prefer explicit, readable code over clever, terse "one-liners." Adhere to the Single Responsibility Principle (SRP) and Interface Segregation Principle (ISP) to ensure the codebase remains malleable and resilient to long-term architectural scaling.


# BACKEND ENGINEERING STANDARDS & PROTOCOLS

1.  **Asynchronous Task Orchestration & Eventual Consistency**
    Decouple high-latency computational logic from the synchronous HTTP request-response cycle using distributed message brokers (e.g., RabbitMQ, Kafka, SQS). Implement robust consumer workers that enforce idempotency to handle "at-least-once" delivery semantics safely. Configure Dead Letter Queues (DLQ) for poison-pill messages and implement exponential backoff strategies for retries to prevent cascading failures during downstream outages.

2.  **Query Plan Optimization & Indexing Strategy**
    Mandate the analysis of database execution plans (`EXPLAIN ANALYZE`) for all complex read operations. Prohibit N+1 query patterns by enforcing eager loading or batched data fetching. Implement composite indices based on access patterns and cardinality; specifically target high-selectivity columns used in `WHERE`, `JOIN`, and `ORDER BY` clauses. Unbounded generic wildcard searches (`LIKE '%term%'`) without full-text search engines (e.g., Elasticsearch) are strictly forbidden.

3.  **Strict HTTP Semantic Compliance & Status Codes**
    Adhere rigidly to RFC 7231 standards for HTTP response codes. Differentiate explicitly between Client Errors (e.g., `400 Bad Request` for syntax, `422 Unprocessable Entity` for semantic validation errors, `401/403` for auth) and Server Errors (`500`). Successful state mutations must return `201 Created` with a `Location` header; idempotent updates utilize `200 OK` or `204 No Content`. Never mask API errors behind a `200 OK` wrapper.

4.  **CORS Governance & Security Headers**
    Implement restrictive Cross-Origin Resource Sharing (CORS) middleware. Prohibit the use of the wildcard `Access-Control-Allow-Origin: *` in production environments; origins must be whitelisted explicitly via configuration. Validate pre-flight `OPTIONS` requests efficiently. Enforce standard security headers including `Strict-Transport-Security` (HSTS), `X-Content-Type-Options: nosniff`, and Content Security Policy (CSP) headers at the gateway level.

5.  **RESTful URI Taxonomy & Versioning**
    Design API endpoints as resource-oriented nouns utilizing hierarchical path structures to represent relationships (e.g., `/resources/:id/sub-resources`). Versioning must be implemented explicitly via URL path prefix (`/api/v1/`) or `Accept` header negotiation to prevent breaking changes. Query parameters should be reserved exclusively for filtering, pagination (`limit`, `offset`/`cursor`), and sorting operations, not for identifying resources.

6.  **Transactional Atomicity & ACID Enforcement**
    Wrap all multi-step mutation operations (CRUD) involving relational data stores within database transactions. Guarantee Atomicity, Consistency, Isolation, and Durability (ACID) properties. Implement optimistic concurrency control via versioning columns (ETags or `lock_version`) to prevent "lost update" anomalies. Rollback logic must be automated and rigorously tested against race conditions.

7.  **Payload Validation & Schema Contracts**
    Enforce strict schema validation on all ingress data (Request Body, Query Params, Path Variables) using declarative validation libraries (e.g., Joi, Zod). Strip unknown fields to prevent mass assignment vulnerabilities. Data serialization and deserialization layers must use DTOs (Data Transfer Objects) to decouple the internal domain model from the public API contract, preventing accidental exposure of sensitive fields (e.g., password hashes, salts).

8.  **Middleware Chain & Interceptor Architecture**
    Architect the request pipeline using a chain-of-responsibility pattern. Separate cross-cutting concerns—authentication, logging, rate-limiting, and context hydration—into distinct, reusable middleware components. Ensure the request context is propagated correctly through the stack without global state pollution. Exception handling must be centralized in a global error boundary that scrubs stack traces before serialization to the client.

9.  **Caching Strategies & Invalidation**
    Implement multi-level caching strategies using high-throughput in-memory stores (Redis/Memcached). Utilize standard HTTP caching headers (`Cache-Control`, `ETag`, `Last-Modified`) for client-side caching. Server-side caching must employ Write-Through or Look-Aside patterns with aggressive Time-To-Live (TTL) settings to balance freshness with latency. Mitigate cache stampedes via locking mechanisms or probabilistic early expiration.

10. **Database Connection Management & Pooling**
    Configure database connection pools with precise limits on minimum/maximum idle connections and connection timeouts to prevent resource exhaustion under load. Implement graceful shutdown procedures to drain active connections properly. Monitor pool saturation and query latency metrics; implement circuit breakers for downstream dependencies to fail fast during infrastructure degradation.