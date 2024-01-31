import grpc
import app_pb2
from app_pb2 import TextRequest, TextResponse
from app_pb2_grpc import PredictionServiceStub  # Corrected import

def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = PredictionServiceStub(channel)  # Corrected instantiation
    text = input("Enter text to predict: ")
    response = stub.PredictText(TextRequest(text=text))
    print("Predicted label:", response.predicted_label)

if __name__ == '__main__':
    main()
