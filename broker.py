import argparse
import json
import logging
import sqlite3
# import signal
import typing

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------

OOM_AUTO_REDUCE_BATCH_SIZE = True


def db_connect(isolation_level: str = None) -> sqlite3.Connection:
    db = sqlite3.connect(
        f'broker.sqlite3',
        check_same_thread=True,
        isolation_level=isolation_level)  # autocommit
    db.execute("PRAGMA journal_mode=WAL")  # concurrency
    return db

# -----------------------------------------------------------------------


app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    path="/api/job",
    description="Get the job list",
    response_class=fastapi.responses.JSONResponse,
)
def getJobList(pageStart: int, count: int, status: str):
    with db_connect() as db:
        if status == '':
            result = db.execute("""
                SELECT id, status, whenQueued, whenStarted, whenEnded, workerHostname, workerGPUID
                FROM jobs
                ORDER BY id DESC
                LIMIT ?, ?
                """, (pageStart, count))
        else:
            result = db.execute("""
                SELECT id, whenQueued, whenStarted, whenEnded, workerHostname, workerGPUID
                FROM jobs
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?, ?
                """, (status, pageStart, count))
        result: sqlite3.Cursor

        return {
            "header": list(map(lambda x: x[0], result.description)),
            "data": result.fetchall(),
        }


@app.get(
    path="/api/job/detail",
    description="Get information of a job",
    response_class=fastapi.responses.JSONResponse,
)
def getJobDetail(jobID: str):
    with db_connect() as db:
        result = db.execute("SELECT * FROM jobs WHERE id == ?", (jobID,))
        result: sqlite3.Cursor

        return dict(zip(
            list(map(lambda x: x[0], result.description)),
            result.fetchall()[0]
        ))


@app.post(
    path="/api/job/control",
    description="CRUD for a job",
    response_class=fastapi.responses.JSONResponse,
)
def setJob(request: str = fastapi.Query(''), payload: dict = fastapi.Body({})):
    with db_connect() as db:
        if request == "insert":
            cur = db.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO jobs (argument, stdin)
                    VALUES (?, ?)
                    """,
                    (payload['argument'], payload['stdin']))
                db.commit()

                jobID = cur.lastrowid
                return {"jobID": jobID}
            except db.Error:
                db.rollback()
                raise fastapi.HTTPException(
                    status_code=fastapi.status.HTTP_404_NOT_FOUND, detail="wrong insert")

        if request == "delete":
            jobID = payload['jobID']
            cur = db.cursor()
            try:
                # incomplete -> NULL
                cur.execute("DELETE FROM jobs WHERE id = ?", (jobID,))
                db.commit()
            except db.Error:
                db.rollback()
                raise fastapi.HTTPException(
                    status_code=fastapi.status.HTTP_404_NOT_FOUND, detail="wrong delete")

            return


def _jobFunctionWorkerSide_requestGet(db: sqlite3.Connection, payload: dict) -> dict:
    # incomplete -> started
    result = db.execute(
        """
            UPDATE jobs
            SET
                status              = "started",
                whenStarted         = datetime('now', 'localtime'),
                workerHostname      = ?,
                workerGPUID         = ?
            WHERE jobs.id = (
                SELECT		id
                FROM		jobs
                WHERE		status = "incomplete"
                ORDER BY	id  ASC
                LIMIT 		1)
            RETURNING jobs.id, jobs.argument, jobs.stdin
        """,
        (payload["workerHostname"], payload["workerGPUID"]),
    ).fetchall()

    if len(result) > 0:
        jobID, argument, stdin = result[0]
        return {"jobID": jobID, "argument": argument, "stdin": stdin}
    else:
        return {"jobID": ""}


def _jobFunctionWorkerSide_requestSet_incomplete(db: sqlite3.Connection, payload: dict):
    jobID, workerHostname, workerGPUID =\
        payload['jobID'], payload['workerHostname'], payload['workerGPUID']

    stdout, stderr = payload['stdout'], payload['stderr']

    # started -> incomplete
    db.execute(
        """
        UPDATE jobs
        SET
            status				= "incomplete",
            whenStarted		    = NULL,
            whenEnded			= NULL,
            workerHostname		= NULL,
            workerGPUID	        = NULL,
            stdout				= ?,
            stderr				= ?
        WHERE
            id				    = ?         AND
            status              = "started" AND
            workerHostname		= ?         AND
            workerGPUID	        = ?
    """,
        (stdout, stderr,
         jobID, workerHostname, workerGPUID))

    return None


def _jobFunctionWorkerSide_requestSet_complete(db: sqlite3.Connection, payload: dict):
    status, jobID, workerHostname, workerGPUID =\
        payload['status'], payload['jobID'], payload['workerHostname'], payload['workerGPUID']

    stdout, stderr = payload['stdout'], payload['stderr']

    # similar to do-while block
    while OOM_AUTO_REDUCE_BATCH_SIZE:
        if status != 'failure':
            break

        if stderr is None:
            break

        if "cuda out of memory" not in stderr.lower():
            break

        stdin = db.execute(
            '''
            SELECT stdin
            FROM jobs
            WHERE id = ? AND status = "started" AND workerHostname = ? AND workerGPUID = ?
            ''',
            (jobID, workerHostname, workerGPUID)).fetchall()[0][0]

        if len(stdin) == 0:
            break

        stdin = json.loads(stdin)

        if 'batchSize' not in stdin:
            break

        if stdin['batchSize'] <= 1:
            # if batchSize is already 1, there is no way to reduce it further
            break

        # stdin['batchSize'] = int(stdin['batchSize'] / 2)
        stdin['batchSize'] = int(float(stdin['batchSize']) * 0.75)
        stdin = json.dumps(stdin)

        # started -> incomplete
        db.execute(
            """
            UPDATE jobs
            SET
                status	            = "incomplete",
                whenStarted		    = NULL,
                whenEnded			= NULL,
                workerHostname		= NULL,
                workerGPUID	        = NULL,
                stdin               = ?,
                stdout				= NULL,
                stderr				= NULL
            WHERE
                id  				= ?         AND
                status				= "started" AND
                workerHostname		= ?         AND
                workerGPUID	        = ?
        """,
            (stdin,
             jobID, workerHostname, workerGPUID))

        return None

    if stdout is not None and len(stdout) == 0:
        stdout = None

    if stderr is not None and len(stderr) == 0:
        stderr = None

    # started -> success, failure
    db.execute(
        """
        UPDATE jobs
        SET
            status		= ?,
            whenEnded	= datetime('now', 'localtime'),
            stdout		= ?,
            stderr		= ?
        WHERE
            id  				= ?         AND
            status				= "started" AND
            workerHostname		= ?         AND
            workerGPUID	        = ?
    """,
        (status, stdout, stderr,
         jobID, workerHostname, workerGPUID))

    return None


@app.post(
    path="/api/job/process",
    description="Process jobs (for Worker)",
    response_class=fastapi.responses.JSONResponse,
)
def jobFunctionWorkerSide(request: str = fastapi.Query(''), payload: dict = fastapi.Body({})):
    with db_connect() as db:
        try:
            # get a incomplete job
            if request == "get":
                return _jobFunctionWorkerSide_requestGet(db, payload)

            # report the job status
            if request == "set":
                status = payload['status']
                if status == 'incomplete':
                    return _jobFunctionWorkerSide_requestSet_incomplete(db, payload)
                elif status == 'success' or status == 'failure':
                    return _jobFunctionWorkerSide_requestSet_complete(db, payload)
        except db.Error:
            db.rollback()
            raise fastapi.HTTPException(status_code=fastapi.status.HTTP_404_NOT_FOUND,
                                        detail="nothing to process (no incomplete job in system)")


# Suppress the "200 OK /api/job/process?request=get" log. Too verbose.
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("POST /api/job/process?request=get") == -1


def main():
    '''
    # due to the db_connect() function, the following code is not necessary
    def _handler(signum: int, _):
        db.close()
        # print(f"DB closed")
        # print(f"Detect {signal.Signals(signum).name}; Goodbye")
        exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='broker_port', default='9999')
    args = vars(parser.parse_args())
    BROKER_PORT = int(args['broker_port'])

    with db_connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id                      INTEGER				PRIMARY KEY AUTOINCREMENT,
                status			    	TEXT				NOT NULL DEFAULT "incomplete",
                whenQueued			    TIMESTAMP			NOT NULL DEFAULT (datetime('now', 'localtime')),
                whenStarted             TIMESTAMP			DEFAULT NULL,
                whenEnded		    	TIMESTAMP			DEFAULT NULL,
                workerHostname          TEXT				DEFAULT NULL,
                workerGPUID     	    UNSIGNED SMALLINT 	DEFAULT NULL,
                argument			    TEXT 				DEFAULT NULL,
                stdin 			    	TEXT 				DEFAULT NULL,
                stdout			    	TEXT 				DEFAULT NULL,
                stderr				    TEXT 				DEFAULT NULL
            )
            """
        )

    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
    uvicorn.run("broker:app", host="0.0.0.0", port=BROKER_PORT)
    # reload=True, reload_dirs=['.'], reload_includes=['broker.py']) # auto reloading is not working as expected; it restarts even when other files are changed


if __name__ == '__main__':
    main()
