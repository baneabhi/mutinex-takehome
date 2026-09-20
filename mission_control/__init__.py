"""Mission Control — a multi-tenant platform for crewing missions.

See docs/DESIGN.md. Section references in docstrings (§6.2, §7.5, ...) point at it.

Layer order, and nothing imports upwards:

    domain      value objects, enums, errors, the eligibility predicate
    authz       permissions, role matrix, contextual policies
    lifecycle   the two transition tables (talks to storage via a Protocol)
    matching    candidate generation, ranking, team assembly, explanation
    solver      the assignment algorithm, no domain knowledge
    store       Platform, Tenant, TenantWorkspace, TenantSnapshot
    services    the operations a caller actually invokes
"""
