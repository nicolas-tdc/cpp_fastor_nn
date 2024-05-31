#pragma once

#include <Fastor/Fastor.h>

#include "Utils/Activator.h"

typedef float Scalar;
typedef Fastor::Tensor<Scalar> BaseTensor;

class BaseLayer {

public:

    // Enumerations.

    enum class Type{Input,
                    Output,
                    Hidden};

    // Constructor and destructor.

    BaseLayer(
        Type layer_type,
        Activator::Type activator_type,
        size_t neurons_count);

    ~BaseLayer();

    // Public properties.

    BaseTensor values;

    BaseTensor weights;

    BaseTensor deltas;

    // Getters and setters.

    // Type.
    void set_type(Type layer_type) { type = layer_type; };
    Type get_type() { return type; };

    // Activator.
    void set_activator(Activator::Type activator_type);
    Activator& get_activator() { return activator; };

private:

    // Private properties.

    Type type;

    Activator& activator;

};