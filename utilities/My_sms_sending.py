
from twilio.rest import Client
class my_SMS_Sending:
  '''for sending notification messages to my personal number'''
# credential for testing api
#   def __init__(self):
#     '''SID, authToke, sending and receving numbers initialization'''
#     self.accountSID = 'ACd23809fdb2dc956703b368240b2aea47'
#     self.authToken = '2f7119da836cbd6ef3220eb20bfaebaa'
#     self.myNumber = '+923159857003'
#     self.twilioNumber = '+15005550006'
  def __init__(self):
    '''SID, authToke, sending and receving numbers initialization'''
    self.accountSID = 'AC3232deea1fbf4186ccb768644a553da8'
    self.authToken = '89db761e457bee24996e508fce75d15f'
    self.myNumber = '+923159857003'
    self.twilioNumber = '+15103302320'
  def send_Message_To_Myself(self, messageText):
    '''-sending sms to my phone number
    +message should be passed as argument for sending'''
    self.messageText = messageText
    self.client = Client(self.accountSID, self.authToken)
    self.messageDetail = self.client.messages.create(
      to=self.myNumber,from_=self.twilioNumber, body=self.messageText)