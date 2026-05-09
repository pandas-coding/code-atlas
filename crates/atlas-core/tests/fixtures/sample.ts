interface User {
    name: string;
    age: number;
}

type ID = string | number;

function getUser(id: ID): User {
    return { name: 'Alice', age: 30 };
}

class Service {
    constructor(private url: string) {}

    async fetch(): Promise<User> {
        return { name: 'Bob', age: 25 };
    }
}
