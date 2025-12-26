# ==============================================================================
# TRANSACTIONS ENDPOINTS - Financial Transaction Routes
# ==============================================================================
# Transaction management and financial analytics endpoints
# ==============================================================================

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import TransactionServiceDep, CurrentUserID
from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.base import APIResponse
from app.schemas.transaction import (
    TransactionCreate,
    TransactionResponse,
    TransactionSummary,
)

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.post(
    "",
    response_model=APIResponse[TransactionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create transaction",
    description="Create a new financial transaction.",
)
async def create_transaction(
    user_id: CurrentUserID,
    schema: TransactionCreate,
    service: TransactionServiceDep,
) -> APIResponse[TransactionResponse]:
    """Create a new transaction."""
    transaction = await service.create_transaction(user_id, schema)
    return APIResponse.ok(data=transaction, message="Transaction created successfully")


@router.get(
    "",
    response_model=APIResponse[List[TransactionResponse]],
    summary="List transactions",
    description="Get all transactions for the current user.",
)
async def list_transactions(
    user_id: CurrentUserID,
    service: TransactionServiceDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    transaction_type: Optional[str] = Query(None, pattern="^(credit|debit|transfer)$"),
    status_filter: Optional[str] = Query(None, alias="status"),
) -> APIResponse[List[TransactionResponse]]:
    """Get user's transactions."""
    transactions = await service.get_user_transactions(
        user_id=user_id,
        skip=skip,
        limit=limit,
        transaction_type=transaction_type,
        status=status_filter,
    )
    return APIResponse.ok(data=transactions)


@router.get(
    "/summary",
    response_model=APIResponse[TransactionSummary],
    summary="Get transaction summary",
    description="Get financial summary for the current user.",
)
async def get_summary(
    user_id: CurrentUserID,
    service: TransactionServiceDep,
) -> APIResponse[TransactionSummary]:
    """Get user's transaction summary."""
    summary = await service.get_user_summary(user_id)
    return APIResponse.ok(data=summary)


@router.get(
    "/{transaction_id}",
    response_model=APIResponse[TransactionResponse],
    summary="Get transaction",
    description="Get a specific transaction by ID.",
)
async def get_transaction(
    transaction_id: str,
    service: TransactionServiceDep,
) -> APIResponse[TransactionResponse]:
    """Get a transaction by ID."""
    try:
        transaction = await service.get_by_id(transaction_id)
        return APIResponse.ok(data=transaction)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found",
        )


@router.get(
    "/reference/{reference_id}",
    response_model=APIResponse[TransactionResponse],
    summary="Get by reference",
    description="Get a transaction by its reference ID.",
)
async def get_by_reference(
    reference_id: str,
    service: TransactionServiceDep,
) -> APIResponse[TransactionResponse]:
    """Get a transaction by reference ID."""
    try:
        transaction = await service.get_by_reference(reference_id)
        return APIResponse.ok(data=transaction)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found",
        )


@router.post(
    "/{transaction_id}/cancel",
    response_model=APIResponse[TransactionResponse],
    summary="Cancel transaction",
    description="Cancel a pending transaction.",
)
async def cancel_transaction(
    transaction_id: str,
    user_id: CurrentUserID,
    service: TransactionServiceDep,
) -> APIResponse[TransactionResponse]:
    """Cancel a pending transaction."""
    try:
        transaction = await service.cancel_transaction(transaction_id, user_id)
        return APIResponse.ok(data=transaction, message="Transaction cancelled")
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found",
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
