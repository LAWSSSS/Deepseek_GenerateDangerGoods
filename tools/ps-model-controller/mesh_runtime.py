"""Bounded subprocess execution and content-addressed 3-D asset caching."""
import os
from pathlib import Path
import subprocess
import time

import psutil


def start_limited(command, stream, memory_bytes, cwd):
    """Start suspended, attach the process tree to a Windows job, then resume."""
    if os.name!="nt":
        return subprocess.Popen([str(x) for x in command],cwd=cwd,stdout=stream,
                                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL),None
    import win32api
    import win32con
    import win32job

    job=win32job.CreateJobObject(None,"")
    limits=win32job.QueryInformationJobObject(job,win32job.JobObjectExtendedLimitInformation)
    limits["BasicLimitInformation"]["LimitFlags"] |= (win32job.JOB_OBJECT_LIMIT_JOB_MEMORY |
                                                       win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    limits["JobMemoryLimit"]=memory_bytes
    win32job.SetInformationJobObject(job,win32job.JobObjectExtendedLimitInformation,limits)
    proc=None
    try:
        proc=subprocess.Popen([str(x) for x in command],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW|0x4)
        handle=win32api.OpenProcess(win32con.PROCESS_SET_QUOTA|win32con.PROCESS_TERMINATE,False,proc.pid)
        try:
            win32job.AssignProcessToJobObject(job,handle)
        finally:
            handle.Close()
        psutil.Process(proc.pid).resume()
        return proc,job
    except Exception:
        if proc is not None:
            proc.kill()
            proc.wait()
        job.Close()
        raise


def run_bounded(command, *, log, seconds, memory_bytes=10*1024**3, cwd=None):
    """Enforce a Windows process-tree memory cap and a wall-clock deadline."""
    start = time.monotonic()
    peak = 0
    with Path(log).open("wb") as stream:
        proc,job = start_limited(command,stream,memory_bytes,cwd)
        root = psutil.Process(proc.pid)
        reason = None
        while proc.poll() is None:
            try:
                family = [root, *root.children(recursive=True)]
                resident = sum(p.memory_info().rss for p in family if p.is_running())
                peak = max(peak, resident)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                resident = 0
            if resident > memory_bytes:
                reason = "memory_limit"
            elif time.monotonic()-start > seconds:
                reason = "timeout"
            if reason:
                try:
                    children = root.children(recursive=True)
                    for child in children:
                        child.kill()
                    root.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                proc.wait(timeout=10)
                break
            time.sleep(.25)
        rc = proc.wait()
        if job is not None:
            job.Close()
    return {"state":reason or ("ok" if rc == 0 else "worker_error"),
            "exit_code":rc, "elapsed_seconds":round(time.monotonic()-start, 2),
            "peak_resident_bytes":peak, "memory_limit_bytes":memory_bytes,
            "memory_limit_mechanism":"windows_job_commit" if os.name=="nt" else "rss_watchdog",
            "timeout_seconds":seconds}


def safe_tail(path, limit=700):
    """Worker diagnostics only, never model authorization headers or API replies."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""
