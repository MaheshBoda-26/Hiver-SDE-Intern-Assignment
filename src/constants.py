"""Intent taxonomy constants for AmazonHelp customer support classification.

These intent labels were derived from manual review of 50-60 sampled AmazonHelp
tweets from the TWCS dataset. They cover the primary categories of customer
inquiries observed in Amazon's Twitter support channel.
"""

INTENTS = [
    "delivery_delay",
    "damaged_defective",
    "refund_request",
    "order_status",
    "account_access",
    "billing_charge",
    "product_question",
    "general_complaint",
    "positive_feedback",
    "other",
]
