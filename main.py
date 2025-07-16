import asyncio
import signal

import grpc
from ai_python_services.proto.ai_service import ai_service_pb2, ai_service_pb2_grpc
from ai_python_services.services.ai_service import AiServiceServicer
from grpc_reflection.v1alpha import reflection


class App:
    """
    Main application class to run the gRPC server.
    """

    def __init__(self, port: int = 50051):
        self.port = port
        self.listen_addr = f"[::]:{port}"
        self.aio_server = None

    async def _setup_server_services(self) -> grpc.aio.Server:
        # Create server inside async context
        self.aio_server = grpc.aio.server()
        self.aio_server.add_insecure_port(self.listen_addr)

        # TODO: ADD SERVICES HERE
        # Add our service to the server
        ai_service_pb2_grpc.add_AiServiceServicer_to_server(AiServiceServicer(), self.aio_server)

        # Enable server reflection
        SERVICE_NAMES = (
            ai_service_pb2.DESCRIPTOR.services_by_name["AiService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, self.aio_server)
        return self.aio_server

    async def shutdown(self):
        """Gracefully shutdown the server."""
        if self.aio_server:
            print("🛑 Shutting down server...")
            await self.aio_server.stop(grace=5)  # Give 5 seconds for graceful shutdown
            print("✅ Server shutdown complete")

    async def run(self):
        """
        Run the gRPC server.
        """
        # Setup server first
        self.aio_server = await self._setup_server_services()

        print(f"🚀 Starting AI gRPC Server (async) on {self.listen_addr}")
        await self.aio_server.start()

        # Setup signal handlers for graceful shutdown
        background_tasks = set()

        def signal_handler():
            print("\n🔄 Received shutdown signal...")
            task = asyncio.create_task(self.shutdown())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, lambda s, f: signal_handler())

        try:
            # Keep the server running
            await self.aio_server.wait_for_termination()
        except KeyboardInterrupt:
            await self.shutdown()
        except Exception as e:
            print(f"❌ Server error: {e}")
            await self.shutdown()


async def main():
    app = App(port=50051)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
