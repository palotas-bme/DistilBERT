# Decorator for removing duplicate values from context fetching / question answering results
def removeduplicates(checkvals):
    def outer(function):
        def inner(*args, **kwargs):
            a = function(*args, **kwargs)
            seen = set()
            unique_vals = []

            for i in a:
                to_check = tuple(i[key] for key in checkvals)
                if to_check not in seen:
                    seen.add(to_check)
                    unique_vals.append(i)

            return unique_vals

        return inner
    return outer