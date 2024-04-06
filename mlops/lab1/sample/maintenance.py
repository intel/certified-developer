# maintenace test business logic

def test_maintenance(
    temperature:int,
    hydraulic_pressure: int,
) -> str:
    """_summary_

    Parameters
    ----------
    temperature : int
    hydraulic_pressure : int
        test parameters for temperature sensor readings

    Returns
    -------
    str
        'Needs Maintenance' or 'No Maintenance Required' based on temperature readings
    """
    if temperature > 50:
        needs = True
    elif hydraulic_pressure > 2000:
        needs = True
    else:
        needs = False

    return 'Needs Maintenance' if needs else 'No Maintenance Required'