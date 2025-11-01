import vonage

client = vonage.Client(key="57915c86", secret="Valet123")
sms = vonage.Sms(client)

response = sms.send_message({
    "from": "13073963461",      # no plus sign for Vonage sender
    "to": "1YOUR_PHONE_NUMBER", # e.g. 18135551234
    "text": "🚗 Test message from Valet System"
})

print(response)
