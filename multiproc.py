import multiprocessing
import time

class multiProc:

    def __init__(self, num_req_p=None, sleep_time=0.1):
        if num_req_p == None:
            num_req_p = multiprocessing.cpu_count()
        self.num_req_p = num_req_p
        print(f"Number of requesterd processes: {self.num_req_p}")
        self.jobs = []
        self.pointer = None
        self.sleep_time = sleep_time
    
    def add_job(self, target_func, target_args):
        p = multiprocessing.Process(target=target_func, args=target_args)
        self.jobs.append(p)

    def add_list_jobs(self, list_jobs):
        for one_job in list_jobs:
            self.add_job(target_func=one_job[0], target_args=one_job[1])
    
    def run_jobs_index(self, i1, i2):
        assert (i2 > i1), f"{i2} should be larger than {i1}"
        self.pointer = i1
        self.job_start_time = time.time() 
        while self.pointer < i2:
            self.check_jobs()
            self.start_job(max_index=i2)
            time.sleep(self.sleep_time)
        self.check_jobs()
        while self.num_running_p > 0:
            self.check_jobs()
            time.sleep(self.sleep_time)
        self.job_elapsed_time = time.time() - self.job_start_time
        self._print_job_info()

    def run_jobs(self):
        self.run_jobs_index(0, len(self.jobs))
    
    def check_jobs(self):
        self.num_running_p = 0
        for proc in self.jobs:
            if proc.is_alive():
                self.num_running_p += 1
    
    def start_job(self, max_index):
        if self.num_running_p < self.num_req_p and self.pointer < max_index:
            print(f"--- START job number {self.pointer}")
            self.jobs[self.pointer].start()
            self.pointer += 1
   
    def __str__(self):
        info = f"#requested processed: {self.num_req_p}"
        info += f"\n#jobs: {len(self.jobs)}"
        return info
    
    def _print_job_info(self):
        print("\n\n" + 10*"=")
        print(f"All jobs finished.")
        print(f"Total Time: {self.job_elapsed_time}")