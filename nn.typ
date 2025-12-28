// Neural Network Drawing Macro for Typst
// Usage: #neural-net(layers: (3, 4, 2), weights: true, ...)

#let neural-net(
  layers: (3, 4, 2),           // Number of neurons in each layer
  neuron-radius: 8pt,          // Radius of neuron circles
  layer-spacing: 80pt,         // Horizontal spacing between layers
  neuron-spacing: 30pt,        // Vertical spacing between neurons
  neuron-fill: white,          // Fill color for neurons
  neuron-stroke: 1pt + black,  // Stroke for neuron circles
  connection-stroke: 0.5pt + gray, // Stroke for connections
  show-weights: false,         // Whether to show weight labels
  weights: none,               // Optional weight matrix array
  weight-decimals: 2,          // Decimal places for weights
  weight-size: 6pt,            // Font size for weight labels
  weight-color: blue,
  label-layers: false,         // Whether to show layer labels
  layer-names: none,           // Custom layer names array
  max-display-neurons: 8,      // Maximum neurons to display before condensing
  condensed-neurons: 4,        // Number of neurons to show at top/bottom when condensed
  always-show-neuron-count: false, // Whether to always show neuron count for all layers
) = {
  
  // Helper function to determine if a layer should be condensed
  let should-condense(layer-size) = {
    layer-size > max-display-neurons
  }
  
  // Helper function to get display neurons for a layer
  let get-display-info(layer-size) = {
    if should-condense(layer-size) {
      (
        display-count: condensed-neurons * 2 + 1, // top neurons + dots + bottom neurons
        actual-count: layer-size,
        is-condensed: true,
      )
    } else {
      (
        display-count: layer-size,
        actual-count: layer-size,
        is-condensed: false,
      )
    }
  }
  
  // Calculate display information for all layers
  let layer-display-info = layers.map(get-display-info)
  
  // Calculate total width and height based on display neurons
  let total-width = (layers.len() - 1) * layer-spacing
  let max-display = calc.max(..layer-display-info.map(info => info.display-count))
  let total-height = (max-display - 1) * neuron-spacing
  
  // Create the neural network diagram
  box(
    width: total-width + 2 * neuron-radius + 20pt,
    height: total-height + 2 * neuron-radius + 60pt,
  )[
    #place(
      dx: neuron-radius + 10pt,
      dy: neuron-radius + 20pt,
    )[
      // Draw connections first (so they appear behind neurons)
      #for l in range(layers.len() - 1) {
        let curr-info = layer-display-info.at(l)
        let next-info = layer-display-info.at(l + 1)
        
        // Calculate vertical centering offsets
        let curr-offset = (max-display - curr-info.display-count) * neuron-spacing / 2
        let next-offset = (max-display - next-info.display-count) * neuron-spacing / 2
        
        // Get positions of visible neurons in current layer
        let curr-positions = ()
        if curr-info.is-condensed {
          // Top neurons
          for i in range(condensed-neurons) {
            curr-positions.push(curr-offset + i * neuron-spacing)
          }
          // Bottom neurons
          for i in range(condensed-neurons) {
            curr-positions.push(curr-offset + (condensed-neurons + 1 + i) * neuron-spacing)
          }
        } else {
          for i in range(curr-info.display-count) {
            curr-positions.push(curr-offset + i * neuron-spacing)
          }
        }
        
        // Get positions of visible neurons in next layer
        let next-positions = ()
        if next-info.is-condensed {
          // Top neurons
          for i in range(condensed-neurons) {
            next-positions.push(next-offset + i * neuron-spacing)
          }
          // Bottom neurons
          for i in range(condensed-neurons) {
            next-positions.push(next-offset + (condensed-neurons + 1 + i) * neuron-spacing)
          }
        } else {
          for i in range(next-info.display-count) {
            next-positions.push(next-offset + i * neuron-spacing)
          }
        }
        
        // Draw connections between all visible neurons
        for start-y in curr-positions {
          for end-y in next-positions {
            let start-x = l * layer-spacing
            let end-x = (l + 1) * layer-spacing
            
            place(
              dx: start-x,
              dy: start-y,
            )[
              #line(
                start: (0pt, 0pt),
                end: (end-x - start-x, end-y - start-y),
                stroke: connection-stroke,
              )
            ]
          }
        }
        
        // Draw weight labels if requested (only for non-condensed layers)
        if show-weights and weights != none and not curr-info.is-condensed and not next-info.is-condensed {
          for i in range(curr-info.display-count) {
            for j in range(next-info.display-count) {
              if l < weights.len() and i < weights.at(l).len() and j < weights.at(l).at(i).len() {
                let w = weights.at(l).at(i).at(j)
                let start-x = l * layer-spacing
                let start-y = curr-offset + i * neuron-spacing
                let end-x = (l + 1) * layer-spacing
                let end-y = next-offset + j * neuron-spacing
                let mid-x = (start-x + end-x) / 2
                let mid-y = (start-y + end-y) / 2
                
                place(
                  dx: mid-x - 8pt,
                  dy: mid-y - 4pt,
                )[
                  #text(size: weight-size, fill: weight-color)[
                    #box(fill: white.transparentize(20%), inset: 1pt, radius: 2pt)[
                      #calc.round(w, digits: weight-decimals)
                    ]
                  ]
                ]
              }
            }
          }
        }
      }
      
      // Draw neurons on top
      #for (l, layer-size) in layers.enumerate() {
        let info = layer-display-info.at(l)
        let layer-offset = (max-display - info.display-count) * neuron-spacing / 2
        
        if info.is-condensed {
          // Draw top neurons
          for i in range(condensed-neurons) {
            let x = l * layer-spacing
            let y = layer-offset + i * neuron-spacing
            
            place(
              dx: x - neuron-radius,
              dy: y - neuron-radius,
            )[
              #circle(
                radius: neuron-radius,
                fill: neuron-fill,
                stroke: neuron-stroke,
              )
            ]
          }
          
          // Draw dots in the middle
          let dots-y = layer-offset + condensed-neurons * neuron-spacing
          place(
            dx: l * layer-spacing - 2pt,
            dy: dots-y - 8pt,
          )[
            #text(size: 20pt, weight: "bold")[⋮]
          ]
          
          // Draw bottom neurons
          for i in range(condensed-neurons) {
            let x = l * layer-spacing
            let y = layer-offset + (condensed-neurons + 1 + i) * neuron-spacing
            
            place(
              dx: x - neuron-radius,
              dy: y - neuron-radius,
            )[
              #circle(
                radius: neuron-radius,
                fill: neuron-fill,
                stroke: neuron-stroke,
              )
            ]
          }
          
          // Draw the count label with curly bracket (always shown for condensed)
          let label-y = layer-offset + info.display-count * neuron-spacing * 0.98 + 0pt
          place(
            dx: l * layer-spacing - 1pt,
            dy: label-y - 2pt,
          )[
            #box[
              #rotate(90deg)[
                #text(size: 16pt)[}]
              ]
            ]
          ]
          place(
            dx: l * layer-spacing - 10pt,
            dy: label-y + 10pt,
          )[
            #box(width: 20pt)[
              #align(center)[
                #text(size: 9pt, weight: "bold")[#layer-size]
              ]
            ]
          ]
          
        } else {
          // Draw all neurons normally
          for i in range(layer-size) {
            let x = l * layer-spacing
            let y = layer-offset + i * neuron-spacing
            
            place(
              dx: x - neuron-radius,
              dy: y - neuron-radius,
            )[
              #circle(
                radius: neuron-radius,
                fill: neuron-fill,
                stroke: neuron-stroke,
              )
            ]
          }
          
          // Show count for non-condensed layers if flag is set
          if always-show-neuron-count {
            let label-y = layer-offset + info.display-count * neuron-spacing + 5pt
            place(
              dx: l * layer-spacing - 1pt,
              dy: label-y - 2pt,
            )[
              #box[
                #rotate(90deg)[
                  #text(size: 16pt)[}]
                ]
              ]
            ]
            place(
              dx: l * layer-spacing - 10pt,
              dy: label-y + 10pt,
            )[
              #box(width: 20pt)[
                #align(center)[
                  #text(size: 9pt, weight: "bold")[#layer-size]
                ]
              ]
            ]
          }
        }
        
        // Draw layer label if requested
        if label-layers {
          let label = if layer-names != none and l < layer-names.len() {
            layer-names.at(l)
          } else if l == 0 {
            "Input"
          } else if l == layers.len() - 1 {
            "Output"
          } else {
            "Hidden " + str(l)
          }
          
          // Place labels above the layers, centered on each layer
          let label-offset = -25pt
          let label-width = if layer-spacing > 60pt { layer-spacing - 10pt } else { 50pt }
          
          place(
            dx: l * layer-spacing - label-width / 2,
            dy: label-offset,
          )[
            #box(width: label-width)[
              #align(center)[
                #text(size: 8pt, weight: "bold")[
                  #label.split("\n").join(linebreak())
                ]
              ]
            ]
          ]
        }
      }
    ]
  ]
}

// CNN Visualization Macro for Typst
// Usage: #cnn-net(layers: (...), ...)

#let cnn-net(
  layers: (),                  // Array of layer specifications
  box-width: 40pt,            // Width of conv/pool boxes
  box-height: 50pt,           // Height of conv/pool boxes
  box-depth: 8pt,             // Depth for 3D effect
  layer-spacing: 30pt,        // Horizontal spacing between layers
  box-stroke: 1pt + black,    // Stroke for boxes
  conv-fill: blue.lighten(70%),
  pool-fill: green.lighten(70%),
  dense-fill: orange.lighten(70%),
  flatten-fill: purple.lighten(80%),
  show-labels: true,          // Show layer labels
  show-dimensions: true,      // Show output dimensions
  label-size: 7pt,
  dimension-size: 6pt,
) = {
  
  // Helper to draw a 3D box
  let draw-3d-box(width, height, depth, fill-color, x, y) = {
    // Front face
    place(dx: x, dy: y)[
      #rect(
        width: width,
        height: height,
        fill: fill-color,
        stroke: box-stroke,
      )
    ]
    
    // Top face (parallelogram)
    place(dx: x, dy: y)[
      #polygon(
        fill: fill-color.darken(10%),
        stroke: box-stroke,
        (0pt, 0pt),
        (width, 0pt),
        (width + depth, -depth),
        (depth, -depth),
      )
    ]
    
    // Right face
    place(dx: x + width, dy: y)[
      #polygon(
        fill: fill-color.darken(20%),
        stroke: box-stroke,
        (0pt, 0pt),
        (depth, -depth),
        (depth, height - depth),
        (0pt, height),
      )
    ]
  }
  
  // Calculate total width
  let total-width = layers.len() * (box-width + layer-spacing)
  
  // Create the diagram
  box(
    width: total-width + 100pt,
    height: 200pt,
  )[
    #let current-x = 20pt
    #let base-y = 100pt
    
    #for (idx, layer) in layers.enumerate() {
      let layer-type = layer.at("type")
      let label = layer.at("label", default: "")
      let dimensions = layer.at("dims", default: none)
      let params = layer.at("params", default: none)
      
      // Determine fill color
      let fill-color = if layer-type == "conv" {
        conv-fill
      } else if layer-type == "pool" {
        pool-fill
      } else if layer-type == "dense" {
        dense-fill
      } else if layer-type == "flatten" {
        flatten-fill
      } else {
        gray.lighten(70%)
      }
      
      // Adjust box size based on layer type
      let current-width = if layer-type == "flatten" {
        15pt
      } else if layer-type == "dense" {
        20pt
      } else {
        box-width
      }
      
      let current-height = if layer-type == "dense" {
        let neuron-count = dimensions.at(0, default: 10)
        30pt + calc.min(neuron-count / 5, 40) * 1pt
      } else if layer-type == "flatten" {
        60pt
      } else {
        box-height
      }
      
      let current-depth = if layer-type == "flatten" {
        3pt
      } else if layer-type == "dense" {
        5pt
      } else {
        box-depth
      }
      
      // Center boxes vertically
      let y-offset = base-y - current-height / 2
      
      // Draw the 3D box
      draw-3d-box(current-width, current-height, current-depth, fill-color, current-x, y-offset)
      
      // Draw arrow to next layer
      if idx < layers.len() - 1 {
        let arrow-start-x = current-x + current-width + current-depth
        let arrow-end-x = arrow-start-x + layer-spacing - current-depth - 5pt
        place(dx: arrow-start-x, dy: base-y - 2pt)[
          #line(
            start: (0pt, 0pt),
            end: (arrow-end-x - arrow-start-x, 0pt),
            stroke: 1pt + gray,
          )
          #place(dx: arrow-end-x - arrow-start-x - 5pt, dy: -3pt)[
            #polygon(
              fill: gray,
              (0pt, 3pt),
              (5pt, 3pt),
              (2.5pt, 6pt),
            )
          ]
        ]
      }
      
      // Draw label above
      if show-labels and label != "" {
        place(dx: current-x + current-width / 2 - 25pt, dy: y-offset - 20pt)[
          #box(width: 50pt)[
            #align(center)[
              #text(size: label-size, weight: "bold")[#label]
            ]
          ]
        ]
      }
      
      // Draw dimensions/params below
      if show-dimensions {
        let info-text = if dimensions != none {
          if layer-type == "conv" or layer-type == "pool" {
            let (h, w, c) = dimensions
            str(h) + "×" + str(w) + "×" + str(c)
          } else if layer-type == "dense" {
            str(dimensions.at(0))
          } else if layer-type == "flatten" {
            str(dimensions.at(0))
          } else {
            ""
          }
        } else {
          ""
        }
        
        let param-text = if params != none {
          "(" + str(params) + ")"
        } else {
          ""
        }
        
        place(dx: current-x + current-width / 2 - 25pt, dy: y-offset + current-height + 5pt)[
          #box(width: 50pt)[
            #align(center)[
              #text(size: dimension-size)[
                #info-text
                #if info-text != "" and param-text != "" [ \ ]
                #text(fill: gray)[#param-text]
              ]
            ]
          ]
        ]
      }
      
      current-x = current-x + current-width + layer-spacing
    }
  ]
}

// Example: Your CNN architecture
#align(center)[
  #cnn-net(
    layers: (
      (type: "conv", label: "Conv2D", dims: (28, 28, 32), params: "3×3"),
      (type: "pool", label: "MaxPool", dims: (14, 14, 32), params: "2×2"),
      (type: "conv", label: "Conv2D", dims: (14, 14, 64), params: "3×3"),
      (type: "pool", label: "MaxPool", dims: (7, 7, 64), params: "2×2"),
      (type: "flatten", label: "Flatten", dims: (3136,)),
      (type: "dense", label: "Dense", dims: (128,)),
      (type: "dense", label: "Output", dims: (10,)),
    ),
    box-width: 45pt,
    box-height: 55pt,
    layer-spacing: 25pt,
    show-labels: true,
    show-dimensions: true,
  )
]

#v(20pt)

// Compact version
#align(center)[
  #cnn-net(
    layers: (
      (type: "conv", label: "Conv2D\n32@3×3", dims: (28, 28, 32)),
      (type: "pool", label: "Pool", dims: (14, 14, 32)),
      (type: "conv", label: "Conv2D\n64@3×3", dims: (14, 14, 64)),
      (type: "pool", label: "Pool", dims: (7, 7, 64)),
      (type: "flatten", label: "Flatten", dims: (3136,)),
      (type: "dense", label: "Dense\n128", dims: (128,)),
      (type: "dense", label: "Output\n10", dims: (10,)),
    ),
    box-width: 40pt,
    box-height: 50pt,
    layer-spacing: 20pt,
    show-dimensions: false,
    conv-fill: blue.lighten(75%),
    pool-fill: green.lighten(75%),
    dense-fill: orange.lighten(75%),
    flatten-fill: purple.lighten(85%),
  )
]