def call_click_wrapper(f, run_params: dict):
    list_of_params = []
    for k, v in run_params.items():
        list_of_params.extend([k, v])
    f(list_of_params)
