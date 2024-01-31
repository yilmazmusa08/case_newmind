import grpc
import app_pb2
import app_pb2_grpc
import uuid
from concurrent import futures
import torch
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from transformers import pipeline

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(String, primary_key=True, nullable=False, unique=True, default=lambda: str(uuid.uuid4()))
    text = Column(String)
    label = Column(String)

class PredictionService(app_pb2_grpc.PredictionServiceServicer):
    def PredictText(self, request, context):
        text = request.text

        # Specify the device
        device = 0  # Use 0 for the first GPU, or 'cuda:0'
        classifier = pipeline("zero-shot-classification",
                            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                            device=device)

        # Print the current device
        print("Current device:", torch.cuda.current_device())

        candidate_labels = ['claim', 'counterclaim', 'evidence', 'rebuttal']

        # Perform classification
        prediction = classifier(text, candidate_labels)

        # Make prediction
        predicted_label = prediction['labels'][0]

        # Store the prediction in the database
        prediction = Prediction(text=text, label=predicted_label)
        session.add(prediction)
        session.commit()

        return app_pb2.TextResponse(predicted_label=predicted_label)

if __name__ == '__main__':
    engine = create_engine('sqlite:///myapp.db')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    app_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionService(), server)
    
    # Change the line below to listen only on localhost
    server.add_insecure_port('localhost:50051')
    
    server.start()

    try:
        print("GRPC Server is running...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
