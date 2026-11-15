"""Agent field names with one release of legacy catalogue compatibility."""


def agent_field(row, name):
    legacy = 'coding_' + name
    if name in row and legacy in row and (
            type(row[name]) is not type(row[legacy]) or row[name] != row[legacy]):
        from .release_contract import ReleaseContractError
        raise ReleaseContractError('conflicting agent field aliases: ' + name)
    return row.get(name, row.get(legacy))


def with_agent_aliases(row):
    """Expose both names without changing the digest-bound source document."""
    result = dict(row)
    for name in ('agent_selected', 'agent_per_cell'):
        legacy = 'coding_' + name
        if name in row or legacy in row:
            result[name] = result[legacy] = agent_field(row, name)
    return result
