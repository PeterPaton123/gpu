import cudnn
print(cudnn.backend_version())
handle: int = cudnn.create_handle()

