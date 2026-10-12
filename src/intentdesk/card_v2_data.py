"""Declared synthetic development augmentation for the separate card workflow.

These are AI-authored examples, not human-reviewed customer records. No check
suite is imported here. Derived variants retain their source family identifier.
"""

import random
import re


SEED = 20261012
ISSUES = {
    "report_lost_card": [
        "I have misplaced my debit card", "Someone stole my bank card",
        "I cannot find my payment card", "My wallet containing my card disappeared",
        "My credit card is missing", "I left my card behind and cannot recover it",
        "Please report the theft of my debit card", "My bank card has gone missing",
    ],
    "report_fraud": [
        "A purchase on my card was not authorized by me", "Someone else used my debit card without permission",
        "There is an unfamiliar card transaction that I did not make", "I want to dispute an unauthorized card charge",
        "I did not approve the payment shown on my card statement", "An unknown merchant charged my credit card",
        "Please investigate suspected fraud on my bank card", "A stranger made a purchase using my card",
    ],
    "damaged_card": [
        "My debit card is cracked", "The chip of my payment card is broken",
        "My credit card snapped in half", "The magnetic stripe on my card is scratched",
        "My card plastic is peeling apart", "The bank card was bent and physically damaged",
        "My card broke into two pieces", "The chip has fallen out of my debit card",
    ],
    "card_declined": [
        "My debit card payment was rejected", "The checkout keeps declining my credit card",
        "The shop will not accept my card payment", "My card transaction failed at the terminal",
        "My payment card was declined at a restaurant", "I have money but my card payment did not go through",
        "The online merchant rejected my card transaction", "The till refused my bank card payment",
    ],
    "pin_change": [
        "I want to change my debit card PIN", "Please help update my bank card PIN number",
        "I need to choose a new PIN for my payment card", "How can I replace my current card PIN",
        "I would like a different PIN on my credit card", "Where can I modify my bank card PIN",
        "I know my card PIN and want to change it", "Please explain how to set a new debit card PIN",
    ],
    "replacement_card_duration": [
        "When will my replacement debit card arrive", "How long is delivery of a replacement bank card",
        "I already ordered a replacement card and need its arrival date", "My replacement credit card has not arrived yet",
        "Please estimate delivery time for my replacement payment card", "How many days does a replacement bank card take to reach me",
        "I am waiting for the replacement debit card in the post", "Where is the replacement card I requested last week",
    ],
}
DENIALS = {
    "report_lost_card": "My card is not lost or stolen",
    "report_fraud": "All the transactions on my card were authorized by me",
    "damaged_card": "My card is not broken or damaged",
    "card_declined": "My card payments have not been declined",
    "pin_change": "I do not want to change my card PIN",
    "replacement_card_duration": "I am not asking about replacement card delivery",
}
OUTSIDE = [
    "I need a new electric kettle", "Help me reserve a railway seat", "My laptop password needs resetting",
    "I broke the SD card in my drone", "My camera memory card is damaged", "I lost a playing card from my deck",
    "My greeting card went missing", "How do I change a SIM card PIN", "I need a new mobile phone unlock PIN",
    "The replacement battery delivery is late", "When will my new shoes arrive", "My gift card was declined",
    "What is the interest rate for a savings account", "Increase my credit card spending limit",
    "Cancel a bank transfer", "Please replace my bank card", "I forgot the PIN for my card",
    "How do I activate a new bank card", "How do I pay my card bill", "Find a nearby cash machine",
    "Help me apply for a mortgage", "Please book a cinema ticket", "Tell me the capital of Peru",
    "Help me cook dinner", "I need a replacement library card", "My employee access card stopped working",
    "I want to change the PIN for my garage keypad", "My travel loyalty card was lost",
    "I want to buy a birthday card", "When will my replacement toothbrush be delivered",
]
VAGUE = [
    "Something is wrong", "Can somebody help me", "It failed again", "Please fix my issue",
    "My bank card needs help", "There is a problem with my payment", "I have a question about my card",
    "What should I do about this", "I cannot explain what happened", "Help with that thing please",
    "I need some card support", "It does not work", "Please sort this out", "I need assistance",
    "My payment has an issue but I have no details", "My card has a problem I cannot describe",
]
CONFLICTS = [
    "My card is lost, or perhaps it is in my pocket; I cannot tell",
    "I said my card was missing but now I think I have it; I am confused",
    "My card is damaged and my card is not damaged; I cannot explain which is true",
    "I cannot tell if this charge is mine or not; I need to check before reporting it",
    "Maybe my payment failed, maybe it succeeded; I have no confirmation",
    "I might want a new PIN but I am not sure what I need",
    "I thought my card was stolen, actually no; I do not know what happened",
    "I cannot decide whether the card is gone or still here",
]


def noisy(text, seed):
    """One reproducible deletion or adjacent transposition, without autocorrection."""
    rng = random.Random(seed)
    words = re.findall(r"\b[A-Za-z]{5,}\b", text)
    if not words:
        return text
    word = rng.choice(words)
    pos = rng.randrange(1, len(word)-1)
    altered = (word[:pos]+word[pos+1]+word[pos]+word[pos+2:] if seed % 2
               else word[:pos]+word[pos+1:])
    return text.replace(word, altered, 1)


def synthetic_rows():
    rows = []
    def add(text, label, family):
        rows.append({"text": text, "label": label, "family": family, "origin": "ai_synthetic_development"})
    for label, phrases in ISSUES.items():
        for i, phrase in enumerate(phrases):
            family = f"single:{label}:{i}"
            for wrapper in ("{}.", "Could you please help? {}.", "This is frustrating! {}!", "After checking my account, {}."):
                add(wrapper.format(phrase), label, family)
            for j in range(3):
                add(noisy(phrase, SEED+i*13+j), label, family)
            for other, denial in DENIALS.items():
                if other != label:
                    add(f"{denial}, but {phrase}.", label, family)
                    add(f"{phrase}. {denial}.", label, family)
    labels = list(ISSUES)
    for a, first in enumerate(labels):
        for second in labels[a+1:]:
            for i in range(8):
                one, two = ISSUES[first][i], ISSUES[second][(i+3) % 8]
                for text in (f"{one}. Separately, {two}. I need help with both.",
                             f"{one}, and I also need help because {two}."):
                    add(text, "uncertain", f"multiple:{first}:{second}:{i}")
    for label, examples in (("oos", OUTSIDE), ("uncertain", VAGUE+CONFLICTS)):
        for i, text in enumerate(examples):
            for prefix in ("", "Please help: ", "Hello, ", "I am frustrated. "):
                add(prefix+text+".", label, f"{label}:{i}")
            add(noisy(text, SEED+i), label, f"{label}:{i}")
    return rows
