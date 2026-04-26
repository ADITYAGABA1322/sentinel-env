from __future__ import annotations

import unittest

from job_queue import GPUJob, JobQueue, JobStatus


class JobQueueTests(unittest.TestCase):
    def test_assign_tick_complete_and_visible_snapshot_hides_priority(self) -> None:
        queue = JobQueue([
            GPUJob(
                job_id="JOB-001",
                priority=5,
                memory_required=40,
                steps_to_complete=2,
                deadline=10,
                owner="team-a",
            )
        ])

        self.assertTrue(queue.assign("JOB-001", "GPU-00"))
        queue.tick(current_step=1, active_job_ids={"JOB-001"})
        self.assertEqual(queue.get("JOB-001").status, JobStatus.RUNNING)
        queue.tick(current_step=2, active_job_ids={"JOB-001"})

        job = queue.get("JOB-001")
        self.assertEqual(job.status, JobStatus.COMPLETE)
        self.assertEqual(job.completed_at, 2)
        self.assertEqual(queue.completion_rate(), 1.0)
        self.assertEqual(queue.deadline_hit_rate(), 1.0)

        visible = queue.snapshot(include_hidden=False)[0]
        hidden = queue.snapshot(include_hidden=True)[0]
        self.assertNotIn("priority", visible)
        self.assertIn("priority", hidden)

    def test_false_completion_only_changes_reported_progress(self) -> None:
        queue = JobQueue([
            GPUJob("JOB-001", priority=3, memory_required=20, steps_to_complete=5, deadline=10, owner="team-a")
        ])
        queue.assign("JOB-001", "GPU-00")

        reward = queue.complete("JOB-001", actual=False, current_step=1)

        self.assertEqual(reward, 0.0)
        self.assertEqual(queue.get("JOB-001").status, JobStatus.RUNNING)
        self.assertEqual(queue.get("JOB-001").actual_progress, 0.0)
        self.assertEqual(queue.get("JOB-001").reported_progress, 1.0)

    def test_deadline_timeout(self) -> None:
        queue = JobQueue([
            GPUJob("JOB-001", priority=2, memory_required=20, steps_to_complete=5, deadline=3, owner="team-a")
        ])

        timed_out = queue.tick(current_step=4)

        self.assertEqual(timed_out, ["JOB-001"])
        self.assertEqual(queue.get("JOB-001").status, JobStatus.TIMED_OUT)


if __name__ == "__main__":
    unittest.main()
