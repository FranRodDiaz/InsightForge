function grafico(json){
    console.log(json)
    var margin = { top: 20, right: 90, bottom: 30, left: 90 },
      width = 960 - margin.left - margin.right,
      height = 500 - margin.top - margin.bottom;

    var svg = d3.select("#tree-svg")
      .attr("width", width + margin.right + margin.left)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svg.style.overflow = "auto";

    var i = 0,
      duration = 750,
      root;
    var treemap = d3.tree().size([height, width]);
    root = d3.hierarchy(json, function(d) {
      return d.children;
    });
    root.x0 = height / 2;
    root.y0 = 0;
    root.children.forEach(collapse);

    update(root);
    function collapse(d) {
      if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
      }
    }

    function update(source) {
      var treeData = treemap(root);
      var nodes = treeData.descendants(),
        links = treeData.descendants().slice(1);
      nodes.forEach(function(d) {
        d.y = d.depth * 180;
      });
      var node = svg.selectAll("g.node").data(nodes, function(d) {
        return d.id || (d.id = ++i);
      });
      var nodeEnter = node
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", function(d) {
          return "translate(" + source.y0 + "," + source.x0 + ")";
        })
        .on("click", click);
      nodeEnter
        .attr("class", "node")
        .attr("r", 10)
        .style("fill", function(d) {
          return d.parent ? "rgb(39, 43, 77)" : "#fe6e9e";
        });
    nodeEnter
        .append("circle")
        .attr("r", 10) // Radio del círculo
        .style("fill", function(d) {
            return d._children ? "lightsteelblue" : "darkgrey"; // Cambia "darkgrey" a cualquier color que prefieras
        })
        .style("stroke", "black") // Agrega un borde negro para mayor visibilidad
        .style("stroke-width", "1.5px");

    nodeEnter
        .append("text")
        .attr("dy", "2em") // Desplazamos el texto debajo del nodo
        .attr("x", function(d) {
            return d.children || d._children ? -13 : 13;
        })
        .style("text-anchor", "middle") // Alineación del texto respecto al nodo
        .text(function(d) { return d.data.name; });


      var nodeUpdate = nodeEnter.merge(node);

      nodeUpdate
        .transition()
        .duration(duration)
        .attr("transform", function(d) {
          return "translate(" + d.y + "," + d.x + ")";
        });
      var nodeExit = node
        .exit()
        .transition()
        .duration(duration)
        .attr("transform", function(d) {
          return "translate(" + source.y + "," + source.x + ")";
        })
        .remove();
      nodeExit.select("rect").style("opacity", 1e-6);
      nodeExit.select("rect").attr("stroke-opacity", 1e-6);
      nodeExit.select("text").style("fill-opacity", 1e-6);
      var link = svg.selectAll("path.link").data(links, function(d) {
        return d.id;
      });
      var linkEnter = link
        .enter()
        .insert("path", "g")
        .attr("class", "link")
        .attr("d", function(d) {
          var o = { x: source.x0, y: source.y0 };
          return diagonal(o, o);
        });
      var linkUpdate = linkEnter.merge(link);
      linkUpdate
        .transition()
        .duration(duration)
        .attr("d", function(d) {
          return diagonal(d, d.parent);
        });
      var linkExit = link
        .exit()
        .transition()
        .duration(duration)
        .attr("d", function(d) {
          var o = { x: source.x, y: source.y };
          return diagonal(o, o);
        })
        .remove();
      nodes.forEach(function(d) {
        d.x0 = d.x;
        d.y0 = d.y;
      });

      var linkText = svg.selectAll("text.link-label")
        .data(links);

      linkText.enter().append("text")
        .attr("class", "link-label")
        .attr("dx", function(d) { return (d.y + d.parent.y) / 2; })
        .attr("dy", function(d) { return (d.x + d.parent.x) / 2; })
        .text(function(d) { return d.data.decision; }) // Aquí asumimos que cada nodo tiene una propiedad 'decision'

    linkText.transition()
        .duration(duration)
        .attr("dx", function(d) { return (d.y + d.parent.y) / 2; })
        .attr("dy", function(d) { return (d.x + d.parent.x) / 2; });

    linkText.exit().remove();


      function diagonal(s, d) {
        path = `M ${s.y} ${s.x}
                C ${(s.y + d.y) / 2} ${s.x},
                  ${(s.y + d.y) / 2} ${d.x},
                  ${d.y} ${d.x}`;

        return path;
      }
      function click(d) {
        if (d.children) {
          d._children = d.children;
          d.children = null;
        } else {
          d.children = d._children;
          d._children = null;
        }
        update(d);
      }
    }
}