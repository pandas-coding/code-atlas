function greet(name) {
    return 'hello ' + name;
}

class Animal {
    constructor(name) {
        this.name = name;
    }

    speak() {
        return this.name + ' makes a noise';
    }
}

const API_URL = 'https://example.com';
