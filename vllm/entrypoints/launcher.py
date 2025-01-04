import asyncio
import signal
from http import HTTPStatus
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response

from vllm import envs
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.multiprocessing import MQEngineDeadError
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.utils import find_process_using_port
from vllm.v1.engine.async_llm import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)


async def serve_http(app: FastAPI, **uvicorn_kwargs: Any):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ', '.join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    watchdog_task = loop.create_task(
        watchdog_loop(server, app.state.engine_client))
    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        watchdog_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port, process, " ".join(process.cmdline()))
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()
    finally:
        watchdog_task.cancel()


async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    # Background task that runs in the background, checking
    # for error state in the engine. This is needed for a
    # clean shutdown since we cannot raise an Exception in
    # a StreamingResponse generator() meaning we cannot use
    # the exception handlers below.
    VLLM_WATCHDOG_TIME_S = 3.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S)
        terminate_if_errored(server, engine)


def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    # See discussions here on shutting down a uvicorn server
    # https://github.com/encode/uvicorn/discussions/1103
    # In this case we cannot await the server shutdown here
    # because handler must first return to close the connection
    # for this request.
    engine_errored = engine.errored and not engine.is_running
    is_already_exiting = server.should_exit
    if (not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored
            and not is_already_exiting):
        # Avoid spamming the logs by only sending once.
        logger.fatal("Engine failed, terminating server.")
        server.should_exit = True


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """
    VLLM V1 AsyncLLM catches exceptions and returns
    only two types: EngineGenerateError and EngineDeadError.
    
    EngineGenerateError is raised by the per request generate()
    method. This error could be request specific (and therefore
    recoverable - e.g. if there is an error in input processing).
    
    EngineDeadError is raised by the background output_handler
    method. This error is global and therefore not recoverable.
    
    We register these @app.exception_handlers to return nice
    responses to the end user if they occur and shut down if needed.
    See https://fastapi.tiangolo.com/tutorial/handling-errors/
    for more details on how exception handlers work.

    NOTE(rob): if an exception is encountered in a StreamingResponse
    generator, the exception is not raised, since we already sent
    a 200 status. Rather, we send an error message as the next chunk.
    Since the exception is not raised, this means that the server
    will not automatically shut down.
    """

    # NOTE(rob): RuntimeError, AsyncEngineDeadError,
    # MQEngineDeadError are all V0 errors.
    @app.exception_handler(RuntimeError)
    @app.exception_handler(AsyncEngineDeadError)
    @app.exception_handler(MQEngineDeadError)
    @app.exception_handler(EngineDeadError)
    @app.exception_handler(EngineGenerateError)
    async def runtime_exception_handler(request: Request, __):
        terminate_if_errored(
            server=server,
            engine=request.app.state.engine_client,
        )

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
