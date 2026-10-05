{#- Safely turn a raw text value such as ' 36 months', '13.99%' or '' into a number. -#}
{% macro to_num(col) -%}
case
    when {{ col }} ~ '[0-9]'
        then regexp_replace({{ col }}, '[^0-9.-]', '', 'g')::double precision
end
{%- endmacro %}

{#- Parse Lending Club month strings like 'Dec-2015'; anything else becomes null. -#}
{% macro to_month(col) -%}
case
    when {{ col }} ~ '^[A-Za-z]{3}-[0-9]{4}$'
        then to_date({{ col }}, 'Mon-YYYY')
end
{%- endmacro %}
