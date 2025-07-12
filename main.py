import asyncio

import grpc
from grpc_reflection.v1alpha import reflection

from ai_python_services.proto import ai_service_pb2, ai_service_pb2_grpc
from ai_python_services.services.ai_service import AiServiceServicer


class App:
    """
    Main application class to run the gRPC server.
    """

    def __init__(self, port: int = 50051):
        self.port = port
        self.listen_addr = f"[::]:{port}"
        # Don't create server here - will be created in async context

    async def _setup_server_services(self) -> grpc.aio.Server:
        # Create server inside async context
        self.aio_server = grpc.aio.server()
        self.aio_server.add_insecure_port(self.listen_addr)

        # TODO: ADD SERVICES HERE
        # Add our service to the server
        ai_service_pb2_grpc.add_AiServiceServicer_to_server(
            AiServiceServicer(), self.aio_server
        )

        # Enable server reflection
        SERVICE_NAMES = (
            ai_service_pb2.DESCRIPTOR.services_by_name["AiService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, self.aio_server)
        return self.aio_server

    async def run(self):
        """
        Run the gRPC server.
        """
        # Setup server first
        self.aio_server = await self._setup_server_services()

        print(f"🚀 Starting AI gRPC Server (async) on {self.listen_addr}")
        await self.aio_server.start()

        try:
            # Keep the server running
            await self.aio_server.wait_for_termination()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            await self.aio_server.stop(0)


async def main():
    app = App(port=50051)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
