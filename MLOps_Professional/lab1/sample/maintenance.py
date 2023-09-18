# maitnenace test business logic

def test_maintenance(temperature:int):
    """_summary_

    Parameters
    ----------
    temperature : int
        test parameter for temperature sensor readings

    Returns
    -------
    str
        'Approved' or 'Denied' based on temperature readings
    """
    maintenance_status = 'Needs Maintenance' if temperature > 50 else 'No Maintenance Required'
    
    return maintenance_status