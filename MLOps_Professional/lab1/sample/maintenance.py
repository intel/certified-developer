# maintenance test business logic


def test_maintenance(temperature: int) -> str:
    """Tests the maintenance status based on temperature sensor readings.

    Args:
        temperature (int): Test parameter for temperature sensor readings.

    Returns:
        str: 'Needs Maintenance' if temperature is greater than 50, otherwise 'No Maintenance Required'.
    """
    maintenance_status = (
        "Needs Maintenance" if temperature > 50 else "No Maintenance Required"
    )

    return maintenance_status
