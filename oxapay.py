"""
OxaPay Payment Integration Module
Handles payment link generation and verification
"""

import aiohttp
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OxaPayAPI:
    """OxaPay API Handler for creating payment links and verifying payments"""
    
    def __init__(self, api_key: str, merchant_api_url: str):
        self.api_key = api_key
        self.merchant_api_url = merchant_api_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def create_payment_invoice(
        self,
        amount: float,
        currency: str = "USD",
        to_currency: str = "USDT",
        order_id: str = None,
        callback_url: str = None,
        description: str = None,
        lifetime: int = 60
    ) -> Dict:
        """
        Create a payment invoice using OxaPay V1 API
        
        Args:
            amount: Payment amount
            currency: Base currency (USD, EUR, etc.)
            to_currency: Target cryptocurrency (USDT, BTC, etc.)
            order_id: Unique order identifier
            callback_url: Webhook URL for payment notifications
            description: Payment description
            lifetime: Invoice lifetime in minutes (default 60)
            
        Returns:
            Dict containing payment link and tracking ID
        """
        try:
            session = await self.get_session()
            
            payload = {
                "amount": amount,
                "currency": currency,
                "lifetime": lifetime,
                "fee_paid_by_payer": 1,
                "under_paid_coverage": 2.5,
                "to_currency": to_currency,
                "auto_withdrawal": False,
                "mixed_payment": True,
                "sandbox": False
            }
            
            if order_id:
                payload["order_id"] = order_id
            
            if callback_url:
                payload["callback_url"] = callback_url
            
            if description:
                payload["description"] = description
            
            headers = {
                "merchant_api_key": self.api_key,
                "Content-Type": "application/json"
            }
            
            logger.info(f"Creating OxaPay invoice for amount: {amount} {currency} -> {to_currency}")
            
            async with session.post(
                "https://api.oxapay.com/v1/payment/invoice",
                json=payload,
                headers=headers
            ) as response:
                result = await response.json()
                
                logger.info(f"OxaPay API response: {result}")
                
                # OxaPay v1 API returns status code in 'status' or 'result' field
                status_ok = (
                    result.get("status") == "success" or 
                    result.get("result") == 100 or
                    response.status == 200
                )
                
                if status_ok and result.get("data"):
                    data = result.get("data", {})
                    payment_data = {
                        "success": True,
                        "payment_link": data.get("payment_url") or data.get("payment_link"),  # API uses payment_url
                        "track_id": data.get("track_id"),
                        "invoice_id": data.get("invoice_id"),
                        "amount": amount,
                        "currency": currency,
                        "order_id": order_id,
                        "message": "Payment invoice created successfully"
                    }
                    logger.info(f"Payment invoice created: {payment_data['track_id']}")
                    return payment_data
                else:
                    error_msg = result.get("message", result.get("error", "Unknown error"))
                    logger.error(f"OxaPay API error - Full response: {result}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "message": f"Failed to create payment invoice: {error_msg}"
                    }
                    
        except Exception as e:
            logger.error(f"Exception in create_payment_invoice: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Error creating payment invoice: {str(e)}"
            }
    
    async def check_payment_status(self, track_id: str) -> Dict:
        """
        Check payment status using track ID
        
        Args:
            track_id: Payment tracking ID
            
        Returns:
            Dict containing payment status information
        """
        try:
            session = await self.get_session()
            
            payload = {
                "merchant": self.api_key,
                "trackId": track_id
            }
            
            logger.info(f"Checking payment status for track_id: {track_id}")
            
            async with session.post(
                "https://api.oxapay.com/merchants/inquiry",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 200 and result.get("result") == 100:
                    status = result.get("status")
                    status_norm = str(status).lower()
                    payment_info = {
                        "success": True,
                        "track_id": track_id,
                        "status": status_norm,
                        "paid": status_norm in ["paid", "confirming"],
                        "confirmed": status_norm == "paid",
                        "amount": result.get("amount"),
                        "currency": result.get("currency"),
                        "pay_amount": result.get("payAmount"),
                        "pay_currency": result.get("payCurrency"),
                        "message": f"Payment status: {status_norm}"
                    }
                    logger.info(f"Payment status checked: {status}")
                    return payment_info
                else:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"OxaPay inquiry error: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "message": f"Failed to check payment status: {error_msg}"
                    }
                    
        except Exception as e:
            logger.error(f"Exception in check_payment_status: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Error checking payment status: {str(e)}"
            }
    
    async def get_white_label_link(
        self,
        amount: float,
        currency: str = "USD",
        pay_currency: str = None,
        order_id: str = None,
        callback_url: str = None,
        description: str = None,
        return_url: str = None
    ) -> Dict:
        """
        Create a white label payment link (allows user to choose cryptocurrency)
        
        Args:
            amount: Payment amount in specified currency
            currency: Fiat currency (USD, EUR, etc.)
            pay_currency: Specific crypto to accept (optional)
            order_id: Unique order identifier
            callback_url: Webhook URL for payment notifications
            description: Payment description
            return_url: URL to redirect after payment
            
        Returns:
            Dict containing payment link and tracking ID
        """
        try:
            session = await self.get_session()
            
            payload = {
                "merchant": self.api_key,
                "amount": amount,
                "currency": currency,
                "lifeTime": 60,  # Link valid for 1 hour
            }
            
            if pay_currency:
                payload["payCurrency"] = pay_currency
            
            if order_id:
                payload["orderId"] = order_id
            
            if callback_url:
                payload["callbackUrl"] = callback_url
            
            if description:
                payload["description"] = description
            
            if return_url:
                payload["returnUrl"] = return_url
            
            logger.info(f"Creating white label payment link for {amount} {currency}")
            
            async with session.post(
                "https://api.oxapay.com/merchants/request/whitelabel",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 200 and result.get("result") == 100:
                    payment_data = {
                        "success": True,
                        "payment_link": result.get("payLink"),
                        "track_id": result.get("trackId"),
                        "amount": amount,
                        "currency": currency,
                        "order_id": order_id,
                        "message": "White label payment link created successfully"
                    }
                    logger.info(f"White label link created: {payment_data['track_id']}")
                    return payment_data
                else:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"OxaPay white label error: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "message": f"Failed to create payment link: {error_msg}"
                    }
                    
        except Exception as e:
            logger.error(f"Exception in get_white_label_link: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Error creating payment link: {str(e)}"
            }
    
    async def revoke_payment(self, track_id: str) -> Dict:
        """
        Revoke/cancel a payment link using track ID
        
        Args:
            track_id: Payment tracking ID
            
        Returns:
            Dict containing revoke status
        """
        try:
            session = await self.get_session()
            
            payload = {
                "merchant": self.api_key,
                "trackId": track_id
            }
            
            logger.info(f"Revoking payment link for track_id: {track_id}")
            
            async with session.post(
                "https://api.oxapay.com/merchants/revoke",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 200 and result.get("result") == 100:
                    logger.info(f"Payment link revoked successfully: {track_id}")
                    return {
                        "success": True,
                        "track_id": track_id,
                        "message": "Payment link revoked successfully"
                    }
                else:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"OxaPay revoke error: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "message": f"Failed to revoke payment: {error_msg}"
                    }
                    
        except Exception as e:
            logger.error(f"Exception in revoke_payment: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Error revoking payment: {str(e)}"
            }
