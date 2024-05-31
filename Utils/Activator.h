#pragma once

class Activator {

public:

    // Enumerations.

    enum class Type {
        Sigmoid,
        ReLu,
        Tanh
    };

    // Constructor and destructor.

    Activator(Type activator_type);

    ~Activator();

    // Processing.

    // Process input with activation function.
    void activate(double& input);

    // Process input with derived activation function.
    double derive(double& input);

    // Getters and setters.

    // Type.
    Type get_type() { return type; };
    void set_type(Type activator_type) { type = activator_type; };

private:

    // Properties.

    Type type;

};